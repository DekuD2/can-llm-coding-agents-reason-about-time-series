import json
import logging
import re
import tempfile
import unittest

from dataclasses import dataclass
from itertools import chain
from litellm import Message
from llm_sandbox.docker import SandboxDockerSession
from llm_sandbox import InteractiveSandboxSession, SandboxSession
from pydantic import BaseModel, Field
from sqlalchemy.types import STRINGTYPE
from uuid import uuid1

from factgenie.annotations import AnnotationModelFactory
from factgenie.colors import Ansi
from factgenie.prompting import transforms as t
from factgenie.prompting.model_apis import MockingAPI, ModelAPI
from factgenie.prompting.text_processing import extract_data

logger = logging.getLogger("factgenie")


@dataclass
class PassData:
    name: str
    data_path: list[str]
    load_command: str = "{{NAME}} = pd.read_json({{PATH}})"


@dataclass
class CallSpec:
    function_name: str
    arguments: list[PassData]
    exec_command: str = """exec_result = {{FUNCTION}}({{ARGS}})
if exec_result is not None:
    print(exec_result)"""


class InterpretCode(t.Transform):
    DEFAULT_SANDBOX_KWARGS = {"keep_template": True, "lang": "python", "default_timeout": 600}

    def __init__(
        self,
        input_field: str,
        output_field: str,
        call_spec: CallSpec,
        sandbox_kwargs: dict = DEFAULT_SANDBOX_KWARGS,
        log_code: bool = False,
    ):
        """
        Args:
            input_field, output_field: The names of input/output field.
            call_spec: A specification of the function to call along with the parameters to send to the sandbox and pass as arguments of the function.
                E.g.:
                ```python
                CallSpec("main",  # The function name
                    [
                        # The first parameter is called 'df'
                        PassData("df",

                                 # The path to the data to send. In this case, it will be in c["data"]["single-dim"]["pandas"], where 'c' referes to the same 'c' as for example the method `DeriveField.apply_function`.
                                 ["data", "single-dim", "pandas"],

                                 # The line of code to load this data.
                                 "{{NAME}} = pd.read_json({{PATH}}); {{NAME}}['time'] = pd.to_datetime({{NAME}}['time'])"
                                )
                    ])
                ```
            sandbox_kwargs: The kwargs passed to the `SandboxSession` constructor.
            log_code: Whether to log the code being executed.
        """
        # This should always be exactly one key (otherwise error should be thrown).
        self.input_field = input_field
        self.output_field = output_field
        self.sandbox_kwargs = sandbox_kwargs
        self.call_spec = call_spec
        self.log_code = log_code
        self.session: SandboxDockerSession = SandboxSession(**self.sandbox_kwargs)
        self.session.open()
        self.libraries = ["numpy", "pandas", "statsmodels", "scipy"]

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field] + [arg.data_path[0] for arg in self.call_spec.arguments]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    def start_interactive(self, c: dict):
        interactive_session = InteractiveSandboxSession(**self.sandbox_kwargs)
        interactive_session.open()

        dst_files = []
        commands = [""]

        for arg in self.call_spec.arguments:
            # Copy the file so to the runtime so it's available under a random temporary name.
            value: str = extract_data(c, arg.data_path)
            random_dst_name = f"/sandbox/{uuid1()}.data"
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                tmp.write(value.encode("utf-8"))
                tmp.flush()  # Really important!!! Otherwise it won't be fully written to the disk and only a part will be copied. (especially for smaller file sizes)
                interactive_session.copy_to_runtime(tmp.name, random_dst_name)
                dst_files.append(random_dst_name)

            random_dst_name_string = f'"{random_dst_name}"'
            commands.append(arg.load_command.replace("{{NAME}}", arg.name).replace("{{PATH}}", random_dst_name_string))

        commands = "\n".join(commands)

        if self.log_code:
            logger.info(f"Starting interactive session with this code: {Ansi.CYAN}{commands}{Ansi.RESET}")

        result = interactive_session.run(commands, libraries=self.libraries)
        if not (result.stderr.isspace() or result.stderr == ""):
            raise Exception(f"Code start error: {Ansi.RED}{result.stderr}{Ansi.RESET}")

        return interactive_session

    def continue_interactive(self, code: str, interactive_session: InteractiveSandboxSession):
        if self.log_code:
            logger.info(f"Running code: {Ansi.CYAN}{code}{Ansi.RESET}")

        result = interactive_session.run(code)
        code_output = result.stdout

        if self.log_code:
            logger.info(f"Code output:\n{Ansi.LIGHT_PURPLE}{code_output}{Ansi.RESET}")
        if not (result.stderr.isspace() or result.stderr == ""):
            logger.error(f"Code error: {Ansi.RED}{result.stderr}{Ansi.RESET}")
            code_output += "\n\n ERROR: \n" + result.stderr

        return code_output

    def interpret(self, c: dict, api: ModelAPI):
        code = c[self.input_field]

        dst_files = []
        commands = [""]

        for arg in self.call_spec.arguments:
            # Copy the file so to the runtime so it's available under a random temporary name.
            value: str = extract_data(c, arg.data_path)
            random_dst_name = f"/sandbox/{uuid1()}.data"
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                tmp.write(value.encode("utf-8"))
                tmp.flush()  # Really important!!! Otherwise it won't be fully written to the disk and only a part will be copied. (especially for smaller file sizes)
                self.session.copy_to_runtime(tmp.name, random_dst_name)
                dst_files.append(random_dst_name)

            random_dst_name_string = f'"{random_dst_name}"'
            commands.append(arg.load_command.replace("{{NAME}}", arg.name).replace("{{PATH}}", random_dst_name_string))

        args = ", ".join(arg.name for arg in self.call_spec.arguments)
        commands.append(
            self.call_spec.exec_command.replace("{{FUNCTION}}", self.call_spec.function_name).replace("{{ARGS}}", args)
        )

        commands = "\n".join(commands)
        full_command = code + "\n" + commands
        if self.log_code:
            logger.info(f"Running code: {Ansi.CYAN}{full_command}{Ansi.RESET}")

        result = self.session.run(full_command, libraries=self.libraries)

        code_output = result.stdout

        if not (result.stderr.isspace() or result.stderr == ""):
            logger.error(f"Code error: {Ansi.RED}{result.stderr}{Ansi.RESET}")
            code_output += "\n\n ERROR: \n" + result.stderr

        # Will be prepended by "[interpreter output]\n" by the agent.
        return code_output

    def __call__(self, current: list[dict], api: ModelAPI):
        return t.derive_field(current, api, self.interpret, self.output_field)


class Edit(t.Transform):
    def __init__(
        self,
        input_field: str,
        output_field: str,
        prepend_value: str = "",
        append_value: str = "",
        prepend_field: str | None = None,
        append_field: str | None = None,
    ):
        assert prepend_value == "" or prepend_field is None
        assert append_value == "" or append_field is None
        self.input_field = input_field
        self.output_field = output_field
        self.prepend_value = prepend_value
        self.prepend_field = prepend_field
        self.append_value = append_value
        self.append_field = append_field

    @property
    def requires_fields(self) -> list[str]:
        return (
            [self.input_field]
            + ([self.prepend_field] if self.prepend_field is not None else [])
            + ([self.append_field] if self.append_field is not None else [])
        )

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    def edit(self, c: dict, api: ModelAPI):
        prepend = self.prepend_value + (c[self.prepend_field] if self.prepend_field is not None else "")
        append = self.append_value + (c[self.append_field] if self.append_field is not None else "")
        return prepend + c[self.input_field] + append

    def __call__(self, current: list[dict], api: ModelAPI):
        return t.derive_field(current, api, self.edit, self.output_field)


class CodingLoop(t.Transform):
    # Inner transforms will be constructed using `CodingLoop.FIELD_...` as their field params.
    FIELD_CONVERSATION = "_CODING_LOOP:CONVERSATION"
    # _FIELD_DECISION = "_CODING_LOOP:DECISION"
    FIELD_CODE_UNEXTRACTED = "_CODING_LOOP:CODE_UNEXTRACTED"
    FIELD_CODE_TO_EXECUTE = "_CODING_LOOP:CODE_TO_EXECUTE"
    FIELD_CODE_OUTPUT = "_CODING_LOOP:CODE_OUTPUT"

    # Not doing so would result in several problems:
    #  - Using arbitrary fields -> big confusion on the side of the user. It becomes like alchemy having to guess what the params should be (or having to read the code to understand the point, which is very complicated).
    #  - Constructing the transforms in this class -> parameter parity problems: having to pass 1 million parameters that have to match the current constructor parameters of the individual transforms. What if we want start_with? start_with_field? system_msg? It becomes too much.

    def __init__(
        self,
        decision_prompt: t.ConverseLLM,
        code_prompt: t.ConverseLLM,
        extract_code: t.ExtractRegex,
        interpret_code: InterpretCode,
        output_field: str,
        max_iters: int = 10,
    ):
        assert (
            decision_prompt.conversation_field == code_prompt.conversation_field
        ), "Coding loop requires its arguments `decision_prompt` and `interpret_code` to share the conversation field."

        self.interpret_code = interpret_code
        self.decision_prompt = decision_prompt
        self.code_prompt = code_prompt
        self.output_field = output_field
        self.max_iters = max_iters

        # It extracts the code from the last response and puts it in the input field for the interpreter.
        self.extract_code = extract_code
        # self.extract_last_as_decision = t.ConversationExtractResponse(self.FIELD_CONVERSATION, self._FIELD_DECISION)
        self.extract_last_as_code_unextracted = t.ConversationExtractResponse(
            self.FIELD_CONVERSATION, self.FIELD_CODE_UNEXTRACTED
        )
        self.prepend_interpreter_tag = Edit(
            self.FIELD_CODE_OUTPUT, self.FIELD_CODE_OUTPUT, prepend_value="[interpreter output]\n"
        )
        self.append_chat = t.ConversationAppendResponse(
            self.FIELD_CONVERSATION, self.FIELD_CODE_OUTPUT, t.ConversationAppendResponse.ROLE_USER
        )

        # So we have 2 `ConverseLLM` transforms, both workign on a common converstaion_field.

        # Decision prompt (-> {"CODE", "ANNOTATE", "ABORT"})
        # Code-writing prompt
        # [Thinking later]
        # Extract code
        # Interpreter

    @property
    def requires_fields(self) -> list[str]:
        required_fields_with_duplicates = chain(self.decision_prompt.requires_fields, self.code_prompt.requires_fields)
        return list(set(required_fields_with_duplicates))

    @property
    def outputs_fields(self) -> list[str]:
        return []

    def evaluate_loop(self, c: dict, api: ModelAPI):
        print("STARTING LOOP")
        # The loop:
        # Ask: What to do... ◄─────────────────────┐
        #   1. CODE?     -> continue loop          │
        #   2. ANNOTATE? -> exit                   │
        #   3. ???       -> ask again              │
        # Ask: CODE_PROMPT -> CODE_RESPONSE        │
        # Extract code block                       │
        # Interpret code                           │
        # Append interpreted code                  │
        # REPEAT ◉─────────────────────────────────┘

        # Copy so we can modify it without consequences.
        c = c.copy()

        for _ in range(self.max_iters):
            # A trick where we:
            #  1. Wrap `c` in a list to make it compatible with the transform.
            #  2. Extract the first element to get a single `c` again.
            c = self.decision_prompt([c], api)[0]
            # c = self.extract_last_as_decision([c], api)[0]
            # logger.info("extract_last", c)

            last_reply = c[self.FIELD_CONVERSATION][-1][t.ConverseLLM.CONTENT]
            if "CODE" in last_reply:
                logger.info("'CODE' outputted")
                # Do a single coding step.
                c = self.code_prompt([c], api)[0]
                c = self.extract_last_as_code_unextracted([c], api)[0]
                c = self.extract_code([c], api)[0]
                c = self.interpret_code([c], api)[0]
                # logger.info(f"FIELD_CODE_UNEXTRACTED: {c[self.FIELD_CODE_UNEXTRACTED]}")
                logger.info(f"CODE RESULT:\n{c[self.FIELD_CODE_OUTPUT]}")
                c = self.prepend_interpreter_tag([c], api)[0]
                c = self.append_chat([c], api)[0]
            elif "ANNOTATE" in last_reply:
                logger.info("'ANNOTATE' outputted")
                # Return the result.
                return c[self.FIELD_CONVERSATION]
            elif "ANSWER" in last_reply:
                logger.info("'ANSWER' outputted")
                # Return the result.
                return c[self.FIELD_CONVERSATION]
            elif "DESCRIBE" in last_reply:
                logger.info("'DESCRIBE' outputted")
                # Return the result.
                return c[self.FIELD_CONVERSATION]
            else:
                logger.info("NOTHING FOUND (gonna ask again)")
                # Try again.
                continue
        logger.info("Ran out of turns...")
        return c[self.FIELD_CONVERSATION]

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return t.derive_field(current, api, self.evaluate_loop, self.output_field)

class AgenticCodingLoop(t.Transform):
    FIELD_CODE_INPUT = "CODING_LOOP:CODE_INPUT"
    FIELD_CODE_OUTPUT = "CODING_LOOP:CODE_INPUT"

    # Not doing so would result in several problems:
    #  - Using arbitrary fields -> big confusion on the side of the user. It becomes like alchemy having to guess what the params should be (or having to read the code to understand the point, which is very complicated).
    #  - Constructing the transforms in this class -> parameter parity problems: having to pass 1 million parameters that have to match the current constructor parameters of the individual transforms. What if we want start_with? start_with_field? system_msg? It becomes too much.

    def __init__(
        self,
        conversation_field: str,
        tool_reply: t.ConverseLLM,
        interpret_code: InterpretCode,
        max_iters: int = 10,  # Alternatively could also make max_tokens.
        # function_name: str = "python",
        function_name: str = "code",
        interactive: bool = False,
        old_description: bool = False,
    ):
        self.conversation_field = conversation_field

        self.interpret_code = interpret_code
        self.tool_reply = tool_reply
        self.max_iters = max_iters
        self.function_name = function_name
        self.interactive = interactive

        self.code_tool = {
            "type": "function",
            "function": {
                "name": function_name,
                "description": "Run code in the interactive python session." if interactive else "Call python interpreter. The function main will be called automatically with the appropriate arguments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The python code to call." if old_description else "The python code to call. Use '\\n' for line breaks.",
                        },
                    },
                    "required": ["code"]
                }
            }
        }

        # self.code_tool = {
        #     "type": "function",
        #     "function": {
        #         "name": "code",
        #         "description": "Call python interpreter. The function main will be called automatically with the appropriate arguments.",
        #         "parameters": {
        #             "type": "string",
        #             "description": "The python code to call.",
        #         }
        #     }
        # }

        self.completion_kwargs = {"tools": [self.code_tool]} # "response_format": self.code_tool  # response format doesn't help. I think I need VLLM.

        # [before]
        # "Answer me this question, you have tools available".
        # -> <conversation>

        # [AgentLoop]
        # The purpose of this transform is to "assist" the LLM by performing its tools until it converges.
        # <conversation> -> extract code call -> <code> -> interpret code -> <result> -> insert tool reply
        # ...
        # <conversation> -> last step is final -> keep it there and continue

        # [after]
        # <conversation> -> extract the answer from it.

    @property
    def requires_fields(self) -> list[str]:
        # HACK: idk why I commented this out and replaced it with the next line.
        # required_fields_with_duplicates = chain(self.tool_reply.requires_fields, [self.conversation_field])
        required_fields_with_duplicates = chain([self.conversation_field])

        return list(set(required_fields_with_duplicates))

    @property
    def outputs_fields(self) -> list[str]:
        return []

    def tool_loop(self, c: dict, api: ModelAPI):
        # The loop:
        #   1. Got conversation as input
        #     2. Conversation got tool call? ◄──┐
        #     3. Extract code                   │
        #     4. Interpret code                 │
        #     5. Reply with tool response ◉─────┘
        #   6. Return last conversation response

        TOOL_CALLS = t.ConverseLLM.TOOL_CALLS
        TYPE = t.ConverseLLM.TYPE
        FUNCTION = t.ConverseLLM.FUNCTION
        NAME = t.ConverseLLM.NAME
        ARGUMENTS = t.ConverseLLM.ARGUMENTS

        # Copy so we can modify it without consequences.
        c = c.copy()

        iters = 0
        self.tool_reply.completion_kwargs = self.completion_kwargs

        interactive_session = None

        while True:
            conv_last: dict = c[self.conversation_field][-1]
            tool_calls: list | None = conv_last.get(TOOL_CALLS, None)

            # As as no tool call is made, we are finished.
            if tool_calls is None:
                break

            # Only create if actually needed because the sandbox starting takes a while.
            if self.interactive and interactive_session == None:
                interactive_session = self.interpret_code.start_interactive(c)

            if len(tool_calls) != 1:
                raise NotImplementedError("We currently don't support more than 1 function call at once.")

            tc = tool_calls[0]
            assert tc[TYPE] == FUNCTION

            func = tc[FUNCTION]
            # logger.warning("START OF TOOL PARSING")
            logger.warning(f"[debug][coding tool] {func[NAME]}:, {tc}")

            # assert func[NAME] == self.function_name

            try:
                # Magistral adds this for some reason
                func[ARGUMENTS] = func[ARGUMENTS].replace("</code>", "")
                try:
                    args: dict = json.loads(func[ARGUMENTS])
                except:
                    args: dict = json.loads(func[ARGUMENTS].replace("\n", "\\n"))
                code = args if isinstance(args, str) else list(args.values())[0]
            except json.JSONDecodeError:
                code = func[ARGUMENTS]

            try:
                if interactive_session:
                    # code = "print(globals())\n" + code # TEMP
                    c[self.FIELD_CODE_OUTPUT] = self.interpret_code.continue_interactive(code, interactive_session)
                else:
                    c[self.FIELD_CODE_INPUT] = code
                    c = self.interpret_code([c], api)[0]

                iters += 1
                if iters == self.max_iters:
                    self.tool_reply.completion_kwargs = {}  # Forget tools.
                    c[self.FIELD_CODE_OUTPUT] += "\n\n[MESSAGE]\n\nMaximum number of tool calls reached. Please submit your answer now."
                    # logger.warning("TOOL warns: Maximum number of tool calls reached. Please submit your answer now.")

                c = self.tool_reply([c], api)[0]
            except Exception as e:
                logger.error(f"Code loop error '{e}'")
                raise e

        if interactive_session:
            interactive_session.close()

        return c[self.conversation_field]

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return t.derive_field(current, api, self.tool_loop, self.conversation_field)


class CustomToolLoop(t.Transform):
    FIELD_CODE_INPUT = "CODING_LOOP:CODE_INPUT"
    FIELD_CODE_OUTPUT = "CODING_LOOP:CODE_INPUT"

    # Not doing so would result in several problems:
    #  - Using arbitrary fields -> big confusion on the side of the user. It becomes like alchemy having to guess what the params should be (or having to read the code to understand the point, which is very complicated).
    #  - Constructing the transforms in this class -> parameter parity problems: having to pass 1 million parameters that have to match the current constructor parameters of the individual transforms. What if we want start_with? start_with_field? system_msg? It becomes too much.

    def __init__(
        self,
        conversation_field: str,
        tool_reply: t.ConverseLLM,
        interpret_code: InterpretCode,
        max_iters: int = 10,  # Alternatively could also make max_tokens.
        function_name: str = "python",
        interactive: bool = False
    ):
        self.conversation_field = conversation_field

        self.interpret_code = interpret_code
        self.tool_reply = tool_reply
        self.max_iters = max_iters
        self.function_name = function_name
        self.interactive = interactive

        self.python_re = r"^\*{0,2}PYTHON:?\*{0,2}:?"
        self.answer_re = r"^\*{0,2}ANSWER:?\*{0,2}:?"

        # [before]
        # "Answer me this question, you have tools available".
        # -> <conversation>

        # [ToolLoop]
        # The purpose of this transform is to "assist" the LLM by performing its tools until it converges.
        # <conversation> -> extract code call -> <code> -> interpret code -> <result> -> insert tool reply
        # ...
        # <conversation> -> last step is final -> keep it there and continue

        # [after]
        # <conversation> -> extract the answer from it.

    @property
    def requires_fields(self) -> list[str]:
        required_fields_with_duplicates = chain(self.tool_reply.requires_fields, [self.conversation_field])
        return list(set(required_fields_with_duplicates))

    @property
    def outputs_fields(self) -> list[str]:
        return []

    def tool_loop(self, c: dict, api: ModelAPI):
        # The loop:
        #   1. Got conversation as input
        #     2. Conversation got tool call? ◄──┐
        #     3. Extract code                   │
        #     4. Interpret code                 │
        #     5. Reply with tool response ◉─────┘
        #   6. Return last conversation response

        # Copy so we can modify it without consequences.
        c = c.copy()

        iters = 0
        interactive_session = None

        while True:
            conv_last: Message = c[self.conversation_field][-1]
            assert isinstance(conv_last.content, str)

            if re.match(self.python_re, conv_last.content):
                code = re.sub(self.python_re, "", conv_last.content)

                if iters < self.max_iters:
                    if interactive_session:
                        c[self.FIELD_CODE_OUTPUT] = self.interpret_code.continue_interactive(code, interactive_session)
                    else:
                        c[self.FIELD_CODE_INPUT] = code
                        c = self.interpret_code([c], api)[0]
                else:
                    c[self.FIELD_CODE_OUTPUT] = "[MESSAGE]\n\nMaximum number of tool calls reached. Please submit your answer now."

                iters += 1
                if iters == self.max_iters:
                    c[self.FIELD_CODE_OUTPUT] += "\n\n[MESSAGE]\n\nMaximum number of tool calls reached. Please submit your answer now."

                c = self.tool_reply([c], api)[0]

                # Only create if actually needed because the sandbox starting takes a while.
                if self.interactive and interactive_session == None:
                    interactive_session = self.interpret_code.start_interactive(c)

            else: #re.match(self.answer_re, conv_last.content):
                if interactive_session:
                    interactive_session.close()

                return c[self.conversation_field]
            # else:
            #     raise ValueError(f"Unexpected answer {conv_last.content}. Expected it to begin with 'PYTHON' or 'ANSWER'.")

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return t.derive_field(current, api, self.tool_loop, self.conversation_field)


class RangeModel(BaseModel):
    start: int = Field(description="The range start.")
    end: int = Field(description="The range end.")

class RangeListModel(BaseModel):
    ranges: list[RangeModel] = Field(description="The list of anomalies.")

class ParseRanges(t.Transform):
    def __init__(
        self,
        input_field: str,
        output_field: str,
    ):
        self.input_field = input_field
        self.output_field = output_field

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    def parse_ranges(self, c: dict, api: ModelAPI):
        try:
            json = c[self.input_field].strip()
            if json[0] == '[':
                json = '{"ranges":' +  json + '}'
            ranges_object = RangeListModel.model_validate_json(json)
            ranges = ranges_object.ranges
            return {self.output_field: [(r.start, r.end) for r in ranges]}
        except:
            logger.warning("Error when parsing ranges, resetting the run...")
            return {"reset_run": True, self.output_field: []}

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return t.derive_and_upsert_fields(current, api, self.parse_ranges)
    

class StringifyRanges(t.Transform):
    def __init__(self, input_field: str, output_field: str):
        """
        Limits the number of entries (to make debugging easier).
        """
        self.input_field = input_field
        self.output_field = output_field

    @property
    def requires_fields(self) -> list[str]:
        return [self.input_field]

    @property
    def outputs_fields(self) -> list[str]:
        return [self.output_field]

    def stringify(self, c: dict, api: ModelAPI):
        ranges = c[self.input_field]
        return str(ranges)

    def __call__(self, current: list[dict], api: ModelAPI):
        return t.derive_field(current, api, self.stringify, self.output_field)


class DataPathAsMetadata(t.Transform):
    def __init__(self, path_to_dict: list[str], new_metadata_name: str):
        self.path_to_dict = path_to_dict
        self.name = new_metadata_name

    # Maybe later create a class 'PassThroughTransform(Transform)' which sets these 3 properties. (Or 'NoTransform' ?)
    @property
    def requires_fields(self) -> list[str]:
        return [self.path_to_dict[0]]

    @property
    def outputs_fields(self) -> list[str]:
        return ["metadata"]

    def with_metadata(self, c: dict):
        previous_metadata = c.get("metadata", {})

        extracted_metadata = extract_data(c, self.path_to_dict)
        return {**c, "metadata": previous_metadata | {self.name: extracted_metadata}}

    def __call__(self, current: list[dict], api: ModelAPI) -> list[dict]:
        return [self.with_metadata(c) for c in current]


class TransformTests(unittest.TestCase):
    THOUGHT = "thinking"

    def __init__(self, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)
        self.api = MockingAPI()
        self.reasoning_api = MockingAPI(include_thought=self.THOUGHT)

    def test_metadata_steal(self):
        current = [{"data": {"old_meta": {"a": "aa", "b": "bb"}}}]
        transform = DataPathAsMetadata(["data", "old_meta"], "stolen_meta")

        expected = [{"data": {"old_meta": {"a": "aa", "b": "bb"}}, "metadata": {"stolen_meta": {"a": "aa", "b": "bb"}}}]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)
    def test_range_parse(self):
        current = [{"json": '{"ranges": [{"start": 3, "end": 55}, {"start": 17, "end": 25}]}'}]
        transform = ParseRanges("json", "ranges")

        expected = [{"json": '{"ranges": [{"start": 3, "end": 55}, {"start": 17, "end": 25}]}',
                     "ranges": [(3, 55), (17, 25)]}]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)

    def test_range_parse_2(self):
        current = [{"json": '[{"start": 3, "end": 55}, {"start": 17, "end": 25}]'}]
        transform = ParseRanges("json", "ranges")

        expected = [{"json": '[{"start": 3, "end": 55}, {"start": 17, "end": 25}]',
                     "ranges": [(3, 55), (17, 25)]}]
        result = transform(current, self.api)

        self.assertListEqual(expected, result)


if __name__ == "__main__":
    logger.disabled = True
    unittest.main()
