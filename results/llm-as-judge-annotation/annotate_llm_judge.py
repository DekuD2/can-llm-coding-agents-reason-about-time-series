#!/usr/bin/env python

import argparse
import concurrent.futures
import dataclasses
import json

# Logging
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from factgenie.llm_judge_model import *

import numpy as np
import orjson
import pyperclip
from rich import print
from rich.rule import Rule
from textual import events
from textual.app import App, ComposeResult, SystemCommand
from textual.containers import Grid, Horizontal, HorizontalScroll, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Collapsible,
    Digits,
    Footer,
    Label,
    ListItem,
    ListView,
    Placeholder,
    Static,
    TextArea,
)

logger = logging.getLogger("app")
logger.setLevel("INFO")
logger.addHandler(logging.FileHandler("app.log"))


# Args
parser = argparse.ArgumentParser()
parser.add_argument("analysis_files", type=str, nargs="+", help="A path to the campaign jsonl file with answers.")
parser.add_argument(
    "--output_file",
    "-o",
    default=None,
    type=str,
    help="File to save to.",
)


class NoMouseWidget(Widget):
    def on_widget_mouse_capture(self, event: events.MouseMove) -> None:
        event.stop()

    def on_widget_mouse_down(self, event: events.MouseMove) -> None:
        event.stop()

    def on_widget_mouse_move(self, event: events.MouseMove) -> None:
        event.stop()

    def on_widget_mouse_release(self, event: events.MouseMove) -> None:
        event.stop()

    def on_widget_mouse_scroll_down(self, event: events.MouseMove) -> None:
        event.stop()

    def on_widget_mouse_scroll_up(self, event: events.MouseMove) -> None:
        event.stop()

    def on_widget_mouse_up(self, event: events.MouseMove) -> None:
        event.stop()


class Header(NoMouseWidget):
    DEFAULT_CSS = """
    Header {
        height: 1;
        dock: top;
    }

    * {
        text-align: center;
    }
    """

    def __init__(self):
        super().__init__()
        self.example_id = "none"
        self.saved = True

    def update_text(self):
        if self.saved:
            # saved = " ([green]all changes saved[/green])"
            saved = ""
        else:
            saved = " ([red]unsaved changes[/red])"
        self.get_child_by_type(Static).content = f"example_id = [bold]{self.example_id}[/bold]{saved}"

    def compose(self) -> ComposeResult:
        yield Static()

    def on_ready(self):
        self.update_text()

    def set_example_id(self, id: int):
        self.example_id = str(id)
        self.update_text()

    def update_saved(self, saved: bool):
        self.saved = saved
        self.update_text()


class Conversation(NoMouseWidget):
    DEFAULT_CSS = """
    MyTextArea {
        # text-wrap: wrap;
        # text-overflow: ellipsis;
        # color: blue;
        height: 1fr;
    }

    Conversation {
        width: 0.55fr;
        dock: left;
    }
    """

    BINDINGS = [
        ("i", "scroll_up", "Scroll up"),
        ("u", "scroll_down", "Scroll down"),
        # ("z", "scroll_exact", "Scroll test"),
        ("g", "scroll_to_start", "Scroll to start"),
        ("G", "scroll_to_end", "Scroll to end"),
        # ("t", "next_thought", "Next thought"),  # Doesn't work well because of wrapping.
        # ("T", "prev_thought", "Prev thought"),  # Doesn't work well because of wrapping.
    ]

    SCROLL_SPEED = 15

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thought_lines = []
        self.tool_lines = []
        self.curr_text = ""

    def compose(self) -> ComposeResult:
        ta = TextArea("conversation...", id="conv_text", read_only=True, show_line_numbers=False)
        ta.styles.width = "100%"
        ta.styles.height = "100%"
        try:
            ta.soft_wrap = True
        except AttributeError:
            pass  # Soft wrap might not be supported in older textual versions
        yield ta

    def action_scroll_down(self) -> None:
        scroll = self.get_child_by_id("conv_text")
        scroll.scroll_relative(y=self.SCROLL_SPEED)

    def action_scroll_up(self) -> None:
        scroll = self.get_child_by_id("conv_text")
        scroll.scroll_relative(y=-self.SCROLL_SPEED)

    def action_next_thought(self) -> None:
        scroll = self.get_child_by_id("conv_text")

        curr_y = scroll.scroll_y
        y = 99999
        for thought_y in reversed(self.thought_lines):
            if thought_y > curr_y:
                y = thought_y
            else:
                break

        scroll.scroll_to(y=y, duration=0.3)

    def action_prev_thought(self) -> None:
        scroll = self.get_child_by_id("conv_text")

        curr_y = scroll.scroll_y
        y = 0
        for thought_y in self.thought_lines:
            if thought_y < curr_y:
                y = thought_y
            else:
                break

        scroll.scroll_to(y=y, duration=0.3)

    def action_scroll_to_start(self) -> None:
        scroll = self.get_child_by_id("conv_text")
        scroll.scroll_to(y=0, duration=0.3)

    def action_scroll_to_end(self) -> None:
        scroll = self.get_child_by_id("conv_text")
        scroll.scroll_to(y=999999, duration=0.3)

    def update_text(self, text: str):
        text = text.replace("\\n", "\n")
        self.curr_text = text

        label = self.get_child_by_id("conv_text")
        with self.app.batch_update():
            label.text = text
            label.scroll_to(y=0, duration=0)

        # # lines = enumerate(text.splitlines())
        # lines = enumerate(label.content.splitlines())
        # self.thought_lines = [i for i, l in lines if "<💭" in l]


class AnnotationItem(NoMouseWidget):
    class EvalChanged(Message):
        pass

    DEFAULT_CSS = """
        AnnotationItem {
            width: 1fr;
            height: auto;
            text-wrap: wrap;
            text-overflow: fold;
        }
    """

    def __init__(self, item: AnnItem, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.item = item

        self.selected_idx = 0
        self.is_selected = False

    def fmt(self, explanation: bool, value, idx: int):
        selected = idx == self.selected_idx and self.is_selected

        def hl(text: str):
            if selected:
                color_if_fg = "#AE5409"  # Orange darker
                color_if_bg = "#F7B178"  # Orange lighter
                bg = len(self.item.eval) == 0
                if bg:
                    color = f"[on {color_if_bg}]"
                    uncolor = f"[/on {color_if_bg}]"
                else:
                    # color = f"[{color_if_fg}]"
                    # uncolor = f"[/{color_if_fg}]"
                    color = ""
                    uncolor = ""
                return f"[underline][bold]{color}" + text + f"{uncolor}[/bold][/underline]"
            else:
                return text

        def eval(text: str):
            yay_color = "#94C7C4"
            nay_color = "#D3B1B5"
            yay_symbol = "✓"
            nay_symbol = ""  # "𐄂"
            if len(self.item.eval) == 0:
                return text
            elif self.item.eval[idx]:
                return f"[on {yay_color}]" + text + yay_symbol + f"[/on {yay_color}]"
            else:
                return f"[on {nay_color}]" + text + nay_symbol + f"[/on {nay_color}]"

        if len(self.item.eval) > 0:
            return hl(eval(str(value)))

        if isinstance(value, bool):
            return f"[green]{hl(eval('True'))}[/green]" if value else f"[red]{hl(eval('False'))}[/red]"
        elif value is None:
            return f"[gray]{hl(eval('None'))}[/gray]"
        elif not explanation:
            return f"[blue]{hl(eval(value))}[/blue]"
        else:
            return f"'{value}'"

    def make_choices_text(self):
        choices = "|".join(self.fmt(False, s, i) for i, s in enumerate(self.item.choices))
        return f"[bold]{self.item.name}[/bold]: {choices}"

    def make_explanation_text(self):
        return f"[bold]{self.item.name}_explanation[/bold]: {self.item.explanations[self.selected_idx]}"

    def compose(self) -> ComposeResult:
        choices_text = self.make_choices_text()
        explanation_text = self.make_explanation_text()
        yield Static(choices_text, id="choices")
        yield Static(explanation_text, id="expl")

    def update_choices(self):
        choices_item: Static = self.get_child_by_id("choices")
        choices_item.content = self.make_choices_text()

    def update_explanation(self):
        expl_item: Static = self.get_child_by_id("expl")
        expl_item.content = self.make_explanation_text()

    def update_selected(self, selected: bool):
        self.is_selected = selected
        self.update_choices()

    def update_selected_item(self, dir: int):
        self.selected_idx += dir
        if self.selected_idx < 0:
            self.selected_idx = len(self.item.choices) - 1
        elif self.selected_idx >= len(self.item.choices):
            self.selected_idx = 0

        self.update_choices()
        self.update_explanation()

    def all_wrong(self):
        # Disallow accidentally marking bools wrong...
        if isinstance(self.item.choices[0], bool):
            first = self.item.choices[0]
            for i in self.item.choices[1:]:
                if i != first:
                    self.notify(
                        "Cannot mark all wrong; [green]True[/green] and [red]False[/red] are both present!",
                        timeout=1.4,
                        severity="warning",
                    )
                    return

        self.item.eval = [False for _ in self.item.choices]
        self.update_choices()
        self.post_message(self.EvalChanged())

    def reset_annotations(self):
        self.item.eval = []
        self.update_choices()
        self.post_message(self.EvalChanged())

    def current_right(self):
        correct = self.item.choices[self.selected_idx]
        self.item.eval = [c == correct for c in self.item.choices]
        self.update_choices()
        self.post_message(self.EvalChanged())


class AnnotationSection(Collapsible):
    DEFAULT_CSS = """
    ListView {
        height: auto;
    }

    ListItem {
        height: auto;
    }
    """

    def __init__(self, category: AnnCategory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_idx = 0
        self.category = category

    def compose(self) -> ComposeResult:
        rows = []
        for i, item in enumerate(self.category.items):
            ann_item = AnnotationItem(item)
            if i == 0:
                ann_item.is_selected = True
            rows.append(ListItem(ann_item))

        yield ListView(*rows, initial_index=None)

    def set_selected(self, idx: int | None = None, dir: int | None = None):
        prev = self.selected_idx

        max_idx = len(self.children[0].children) - 1
        if dir is not None:
            self.selected_idx += dir
            if self.selected_idx > max_idx:
                self.selected_idx = 0
            elif self.selected_idx < 0:
                self.selected_idx = max_idx
        elif idx is not None:
            self.selected_idx = idx
        else:
            raise ValueError("Either idx or dir has to be specified (`get_selected`).")

        def update_child(idx: int, selected: bool):
            list_view = self.get_child_by_type(ListView)
            list_item: ListItem = list_view.children[idx]
            annotation_item = list_item.get_child_by_type(AnnotationItem)
            annotation_item.update_selected(selected)
            list_item.highlighted = False

        update_child(prev, False)
        update_child(self.selected_idx, True)

    def select_lr(self, dir: int):
        list_view = self.get_child_by_type(ListView)
        list_item: ListItem = list_view.children[self.selected_idx]
        annotation_item = list_item.get_child_by_type(AnnotationItem)
        annotation_item.update_selected_item(dir=dir)

    def eval(self, yay: bool | None):
        list_view = self.get_child_by_type(ListView)
        list_item: ListItem = list_view.children[self.selected_idx]
        annotation_item = list_item.get_child_by_type(AnnotationItem)

        if yay == True:
            annotation_item.current_right()
        elif yay == False:
            annotation_item.all_wrong()
        elif yay is None:
            annotation_item.reset_annotations()


class Annotation(NoMouseWidget):
    DEFAULT_CSS = """
    Annotation {
        width: 0.45fr; # or 50%
        dock: right;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_section = 0

    def compose(self) -> ComposeResult:
        # with VerticalScroll(id="ann_scroll"):
        #     yield Static("annotations...", id="ann_text")
        # with Collapsible():
        yield Static("annotations...", id="ann_text")
        yield Static("annotations...")
        yield Static("annotations...")

    def get_sections(self, text: str):
        sections = {}
        sections["categories"] = {}
        j = json.loads(text)
        for k, v in j.items():
            if isinstance(v, dict):
                sections[k] = v
            else:
                sections["categories"][k] = v
        return sections

    def get_curr_section(self) -> AnnotationSection:
        collapsible = self.children[self.selected_section]
        assert isinstance(collapsible, Collapsible)
        # logger.info("children: " + str(content.children) + " of types " + ", ".join(map(str, map(type, content.children))))
        annotation_section = collapsible.children[1].get_child_by_type(AnnotationSection)
        return annotation_section

    def nav(self, dir: str):
        section = self.get_curr_section()
        if dir == "u":
            section.set_selected(dir=-1)
        elif dir == "d":
            section.set_selected(dir=1)
        elif dir == "l":
            section.select_lr(-1)
        elif dir == "r":
            section.select_lr(1)

    def eval(self, yay: bool | None):
        self.get_curr_section().eval(yay)

    def update_text(self, annotations: Ann):
        with self.app.batch_update():
            self.remove_children()
            i = 0
            widgets = []
            for cat in annotations.categories:
                widgets.append(
                    Collapsible(AnnotationSection(cat), title=cat.name, collapsed=i != self.selected_section)
                )
                i += 1
            self.mount(*widgets)

    def change_section(self, dir=1) -> None:
        n_children = len(self.children)
        self.selected_section += dir
        if self.selected_section >= n_children:
            self.selected_section = 0
        elif self.selected_section < 0:
            self.selected_section = n_children - 1

        for i, c in enumerate(self.children):
            if isinstance(c, Collapsible):
                c.collapsed = i != self.selected_section


class ConvAnn(NoMouseWidget):
    BINDINGS = [
        ("m", "change_section", "Next section"),
        ("M", "change_section_back", "Prev section"),
        ("j", "ann_down", "↓ ann."),
        ("k", "ann_up", "↑ ann."),
        ("h", "ann_left", "<- ann."),
        ("l", "ann_right", "-> ann."),
        ("enter", "ann_yay", "Mark correct"),
        ("backspace", "ann_none", "Mark all wrong"),
        ("q", "ann_none", "Mark all wrong"),
        ("=", "ann_reset", "Unmark"),
    ]

    def compose(self) -> ComposeResult:
        yield Conversation()
        yield Annotation()

    def update_dataset_item(self, ds_item: DatasetItem):
        self.get_child_by_type(Conversation).update_text(ds_item.conversation)
        self.get_child_by_type(Annotation).update_text(ds_item.annotations)

    def action_change_section(self) -> None:
        self.get_child_by_type(Annotation).change_section(1)

    def action_change_section_back(self) -> None:
        self.get_child_by_type(Annotation).change_section(-1)

    def action_ann_down(self) -> None:
        self.get_child_by_type(Annotation).nav("d")

    def action_ann_up(self) -> None:
        self.get_child_by_type(Annotation).nav("u")

    def action_ann_left(self) -> None:
        self.get_child_by_type(Annotation).nav("l")

    def action_ann_right(self) -> None:
        self.get_child_by_type(Annotation).nav("r")

    def action_ann_yay(self) -> None:
        self.get_child_by_type(Annotation).eval(True)

    def action_ann_none(self) -> None:
        self.get_child_by_type(Annotation).eval(False)

    def action_ann_reset(self) -> None:
        self.get_child_by_type(Annotation).eval(None)


class GoToIdTextArea(TextArea):
    def __init__(self, modal, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modal = modal

    def _on_key(self, event: events.Key) -> None:
        if event.name == "enter":
            try:
                number = int(self.text.strip())
                self.modal.dismiss(number)
            except:
                self.modal.dismiss(None)

        if isinstance(event.character, str) and not event.character.isdigit():
            event.prevent_default()


class GoToId(ModalScreen):
    CSS = """
        Grid {
            width: 20;
            height: 7;
            align: center middle;
        }

        #g1 {
            grid-size: 1 2;
        }

        #g2 {
            grid-size: 2 1;
        }
    """

    # def __init__(self, annotation_app, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.annotation_app = annotation_app

    def compose(self) -> ComposeResult:
        with Grid(id="g1"):
            # yield TextArea()
            yield GoToIdTextArea(self)
            with Grid(id="g2"):
                yield Button("Go to id", variant="primary", id="goto")
                yield Button("Cancel", variant="error", id="quit")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "goto":
            text = self.get_child_by_type(Grid).get_child_by_type(GoToIdTextArea).text.strip()
            try:
                number = int(text)
                self.dismiss(number)
            except:
                self.dismiss(None)
        else:
            self.dismiss(None)


class AnnotationApp(App):
    CSS = """
    Screen { align: center middle; }
    """

    BINDINGS = [
        ("n", "next_example", "Next example"),
        ("N", "prev_example", "Prev example"),
        ("d", "toggle_dark", "Toggle dark mode"),
        ("ctrl+s", "save", "Save"),
    ]

    def __init__(self, dataset_evaluation: DatasetEval, savename: str):
        super().__init__()
        self.dataset_evaluation = dataset_evaluation
        self.dataset_items = dataset_evaluation.examples
        self.dataset_item_id = -1
        self.savename = savename
        self.savename_tmp = savename + ".tmp"
        self.update_needed = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield ConvAnn()
        yield Footer()

    def go_to_example(self):
        def goto(id: int | None):
            if isinstance(id, int):
                self.update_example(set_id=id)

        self.push_screen(GoToId(), goto)

    def on_annotation_item_eval_changed(self, event: AnnotationItem.EvalChanged):
        self.update_needed = True
        self.get_child_by_type(Header).update_saved(False)

    def save(self, manual: bool = False):
        with open(self.savename_tmp, "wb") as f:
            f.write(self.dataset_evaluation.to_json())

        # Atomic move (because we are on the same file system) once the save is succesfully completed.
        shutil.move(self.savename_tmp, self.savename)

        self.update_needed = False
        self.get_child_by_type(Header).update_saved(True)

        if manual:
            timeout = 0.8
        else:
            timeout = 2.2

        self.notify("💾 Saved", timeout=timeout, severity="information")

    def action_save(self):
        self.save(manual=True)

    def copy_example_to_clipboard(self):
        text = self.get_child_by_type(ConvAnn).get_child_by_type(Conversation).curr_text
        pyperclip.copy(text)

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        yield SystemCommand("Go to example", "Go to example by id", self.go_to_example)
        yield SystemCommand("Save", f"Save the current state to '{self.savename}'", self.save)
        yield SystemCommand("Clipboard", f"Copy the conversation to clipboard", self.copy_example_to_clipboard)

    def on_ready(self) -> None:
        self.action_next_example()
        self.action_toggle_dark()
        # self.set_interval(1, self.update_clock)

    def update_example(self, dir: int = 1, set_id: int | None = None):
        self.dataset_item_id += dir
        if self.dataset_item_id >= len(self.dataset_items):
            self.dataset_item_id = 0
        if self.dataset_item_id < 0:
            self.dataset_item_id = len(self.dataset_items) - 1

        if set_id:
            self.dataset_item_id = set_id
            if self.dataset_item_id >= len(self.dataset_items):
                self.dataset_item_id = len(self.dataset_items) - 1
            if self.dataset_item_id < 0:
                self.dataset_item_id = 0

        ds_item = self.dataset_items[self.dataset_item_id]

        self.get_child_by_type(ConvAnn).update_dataset_item(ds_item)
        self.get_child_by_type(Header).set_example_id(self.dataset_item_id)

    def save_if_changed(self):
        if self.update_needed:
            self.save()

    def action_next_example(self) -> None:
        self.update_example(dir=1)
        self.save_if_changed()

    def action_prev_example(self) -> None:
        self.update_example(dir=-1)
        self.save_if_changed()

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.theme = "textual-dark" if self.theme == "textual-light" else "textual-light"


def merge_files(analyses: list[list[FileProxy]]):
    def merged_analyses_at_index(i: int):
        curr_idx_examples = [analyses[j][i] for j in range(len(analyses))]
        return FileProxy(
            conversation=curr_idx_examples[0].conversation,
            annotations=[curr_idx_examples[j].annotations[0] for j in range(len(analyses))],
        )

    min_len = max(map(len, analyses))
    for i in range(min_len):
        # For each example `i` (2nd index), check that all analyses are about the same conversation, and that they have just one annotation so far.
        for j in range(0, len(analyses)):
            if j > 0:
                assert (
                    analyses[0][i].conversation == analyses[j][i].conversation
                ), f"Conversations in example index {i} don't match!"  # WARNING: This might be slow?
            assert (
                len(analyses[j][i].annotations) == 1
            ), "Can only merge non-merged annotations"  # INFO: Would be possible to implement but no reason to.

    return [merged_analyses_at_index(i) for i in range(min_len)]

def main(args):
    input_files = args.analysis_files
    if len(input_files) == 1 and input_files[0].name.endswith(".json"):
        first: str = input_files[0].name
        with open(first, "r") as f:
            dataset_eval = DatasetEval.from_json(f.read())
        savename = str(args.analysis_files[0])
    else:
        ds_infos, evaluators = map(list, zip(*(get_metadata(analysis_file) for analysis_file in input_files)))
        assert all(
            ds_infos[0] == ds_infos[i] for i in range(1, len(ds_infos))
        ), "Dataset infos don't match between files!"

        files = [load_file(analysis_file) for analysis_file in input_files]
        merged = merge_files(files)  # Checks that conversations are identical
        dataset_items = [DatasetItem(i, m.conversation.strip(), Ann(m.annotations)) for i, m in enumerate(merged)]

        dataset_eval = DatasetEval(ds_infos[0], evaluators, dataset_items)

        if args.output_file is not None:
            savename = args.output_file
        else:
            savename = "default_save.json"

        if not savename.endswith(".json"):
            savename = savename + ".json"

        if Path(savename).exists():
            print(
                f"'{str(savename)}' already exists. You have two options:\n - Specify different `--output_file` argument.\n - Use the script in the following form: `{Path(__file__).name} {str(savename)}`"
            )
            exit(1)

        print(f"Save file will be named '{savename}'...")

    # Pre-parse the JSON annotations for all examples so navigating is fast
    print("Pre-parsing annotations...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(lambda ds: ds.annotations.categories, dataset_eval.examples))

    app = AnnotationApp(dataset_eval, savename)
    app.run()


def update_path(args, arg_name):
    orig_val = getattr(args, arg_name, None)
    if orig_val is None:
        return args

    def update_path(path: Path):
        if path.is_dir():
            if (path / "files").exists():
                path = path / "files"

            if (path / "combined.jsonl").exists():
                path = path / "combined.jsonl"
            else:
                files = list(path.glob("*.jsonl"))
                if len(files) == 1:
                    path = files[0]
                else:
                    print(f"Too many options in '{path}'.")
                    for f in files:
                        print(f" - '{str(f)}'")
                    raise ArgumentError("Too many options in '{path}'.")
        return path

    paths = [update_path(Path(p)) for p in getattr(args, arg_name)]

    setattr(args, arg_name, paths)
    # if str(orig_val) != str(paths):
    #     print(f"updating '{orig_val}' -> '{paths}'")
    return args


if __name__ == "__main__":
    args = parser.parse_args()
    for arg_name in ["analysis_files"]:
        args = update_path(args, arg_name)

    main(args)
