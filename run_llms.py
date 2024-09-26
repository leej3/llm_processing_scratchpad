from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import openai
import os
import json
import pickle
import logging
import traceback
from lxml import etree
import time
from datetime import datetime
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )


def remove_references(xml_input: str) -> str:
    """
    Removes the <div type="references"> section from a TEI XML string.

    :param xml_input: A string containing the original TEI XML.
    :return: A string of the modified TEI XML without the references section.
             Returns None if an error occurs during processing.
    """
    try:
        # Define the TEI namespace
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        # Parse the XML string
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(xml_input.encode('utf-8'), parser)

        # Find all <div> elements with type="references"
        references_divs = root.findall('.//tei:div[@type="references"]', namespaces=ns)

        if not references_divs:
            logger.info("No references section found in the XML.")
            return etree.tostring(root, pretty_print=True, encoding='UTF-8', xml_declaration=True).decode('utf-8')

        for div in references_divs:
            parent = div.getparent()
            if parent is not None:
                parent.remove(div)
                logger.info("Removed a <div type=\"references\"> section.")

        # Convert the modified XML tree back to a string
        modified_xml = etree.tostring(root, pretty_print=True, encoding='UTF-8', xml_declaration=True).decode('utf-8')
        return modified_xml

    except etree.XMLSyntaxError as e:
        logger.warning(f"XML Syntax Error: {e}")
        return xml_input
    except Exception as e:
        logger.warning(f"An unexpected error occurred: {e}")
        return xml_input

class LLMExtractorMetrics(BaseModel):
    """
    Model for extracting information from scientific publications. These metrics
    are a summary of the publications adherence to transparent or open
    scientific practices. With regards to code and data this implies that the
    corresponding artifact has been shared, not "will be shared" or shared "upon
    request". "Will be shared", "by request", or "upon request" is a statement
    implying lack of sharing of the corresponding study artifacts. An
    inaccessible sharing statement in the appendix or supplmentary materials is
    likely a statement of successful sharing and should imply True for the
    corresponding boolean field.

    Fields for which statements are reported should be taken from the input
    verbatim without concern about the resulting grammatical incorrectness that
    might occurring from only requiring part of a sentence etc.
    """

    model: str = Field(
        description="Exact verion of the llm model used to generate the data (not in publication itself but known by the model) e.g. GPT_4o_2024_08_06"
    )
    year: int = Field(
        description="Best attempt at extracting the year of the publication or use the int 9999",
    )
    country: list[str] = Field(
        description="The countries of the affiliations of the authors",
    )
    institute: list[str] = Field(
        description="The institutes of the affiliations of the authors",
    )
    code_sharing_statement: list[str] = Field(
        description="Statements in the paper that indicate that the code used for analysis has been shared freely online or has not been made available.",
    )
    code_repository_url: str = Field(
        description="The URL of the repository where the code and data can be found if it is included."
    )
    is_open_code: bool = Field(
        description="Whether there is evidence that the code used for analysis in the paper has been shared online. The fields code_sharing_statement and code_repository_url should be used to determine this.",
    )
    data_sharing_statement: list[str] = Field(
        description="Statements in the paper that indicate that the data used for analysis has been shared freely online or has not been made available.",
    )
    data_repository_url: str = Field(
        description="The URL of the repository where the data can be found if it is provided"
    )
    dataset_unique_identifier: list[str] = Field(
        description="Any unique identifiers the dataset may have"
    )
    is_open_data: bool = Field(
        description="Whether there is evidence that the data used for analysis in the paper has been shared online. The fields data_sharing_statment data_repository_url and dataset_unique_identifier should be used to determine this.",
    )
    reasoning_steps: list[str] = Field(
        description="The reasoning steps used to extract the information from the paper. This can verbosely provide context or explanation in the decision making process. Be verbose for this field. Do not leave empty. Each statement should be surrounded by double quotes and use backslash to escape any double quote in the statements themselves. Do not use tabs or newline characters at all.",
    )


def get_initial_message(model: str, xml_content: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are an expert at extracting information from scientific publications with a keen eye for details that when combined together allows you to summarize aspects of the publication",
        },
        {
            "role": "user",
            "content": (
                f"The llm model is {model}. Please extract in accordance with the schema provided. The publication in xml follows below:\n"
                "------\n"
                f"{xml_content}\n"
                "------"
            ),
        }
    ]
def extract_using_model(xml_content: str, model: str) -> LLMExtractorMetrics:
    messages = get_initial_message(model, xml_content)
    for attempt in range(6):
        messages, result = attempt_extraction(messages, model)
        if result is not None:
            return messages, result
    raise ValueError(f"Failed to extract information from the publication: \n\n {messages[2:]} \n\n")

def attempt_extraction(messages: list[dict], model: str) -> None:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[
                # strict=True is set by this helper method
                openai.pydantic_function_tool(LLMExtractorMetrics),
            ],
        )
        try:
            response_message = completion.choices[0].message
            tool_calls = response_message.tool_calls
            messages.append(response_message)
        except TypeError as e:
            if "NoneType" in str(e):
                messages.append({
                    "role":"user",
                    "content":f"{e} \n\n That last attempt seemed to have no output. Try to populate the schema correctly...",
                })
                return messages, None
            else:
                raise e
    except Exception:
        err = traceback.format_exc()
        messages.append({
            "role":"user",
            "content":f"{err} \n\n That last attempt resulted in the above error. Can we try again...",
        })
        return messages, None
    if not tool_calls:
        messages.append({
            "role":"user",
            "content":"That doesn't have a tool call. Can we try again...",
        })
        return messages, None
    else:
        # If true the model will return the name of the tool / function to call and the argument(s)
        tool_call_id = tool_calls[0].id
        tool_function_name = tool_calls[0].function.name

        if tool_function_name == 'LLMExtractorMetrics':
            try:
                args_dict = json.loads(tool_calls[0].function.arguments)
                result = LLMExtractorMetrics(**args_dict)
                return messages, result
            except Exception as e:
                messages.append({
                    "role":"tool",
                    "tool_call_id":tool_call_id,
                    "name": tool_function_name,
                    "content":e,
                })
                return messages, None


def main():
    model = "openai/gpt-4o-mini"
    # model = "anthropic/claude-3.5-sonnet"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filepath = Path(f"tempdata/llm_extractions/{model.replace("/","-")}_{timestamp}.feather")
    print("output_filepath:", output_filepath)
    if not output_filepath.parent.exists():
        output_filepath.parent.mkdir(parents=True)

    df = pd.read_feather("tempdata/combined_metadata.feather")
    # Perform the 90/10 split
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    if os.environ.get("VALIDATE", False):
        pipeline_df = test_df
    else:
        pipeline_df = train_df
    with_xml = (
        pipeline_df
        .sort_index()
        .assign(
            xml_path=lambda df: df.filename.str.replace("combined_pdfs", "full_texts").str.replace(".pdf", ".xml"),
            xml=lambda x: x.xml_path.apply(lambda y: Path(y).read_text()),
            xml_for_llm=lambda x: x.xml.apply(remove_references),
        )
    )

    outputs = []
    for idx, row in with_xml.iterrows():
        print(f"Processing row {idx}")
        xml_content = row.xml_for_llm
        try:
            messages, result = extract_using_model(xml_content, model)
            metrics = result.model_dump(mode="json")
            metrics['idx'] = idx
            outputs.append(metrics)
            # store intermediate results and chat history
            output_filepath.with_suffix(".pkl").write_bytes(pickle.dumps(outputs))
            messagesdir = output_filepath.parent/ output_filepath.stem
            messagesdir.mkdir(exist_ok=True)
            (messagesdir / f"{idx}.pkl").write_bytes(pickle.dumps(messages))
        except Exception:
            err = traceback.format_exc()
            with open(output_filepath.with_suffix(".err"), "a") as log_file:
                log_file.write(f"Error processing row {idx}: {err}\n\n")
            logger.warning(f"Error processing row {idx}: {err[-300:]}...")
    df_llm = pd.DataFrame(outputs).set_index('idx').assign(reasoning_steps=lambda x: x["reasoning_steps"].astype(str))
    df_out = df_llm.rename(columns={col: f"llm_{col}" for col in df_llm.columns}).join(pipeline_df)
    (
        # save the mismatched predictions to a tsv
        df_out
        .query("manual_is_open_data != llm_is_open_data")
        [["manual_is_open_data","llm_is_open_data","llm_data_sharing_statement","manual_data_statements","doi","filename","llm_reasoning_steps"]]
        .to_csv(str(output_filepath).replace(".feather","_misses.tsv"),sep="\t",index=False)
    )
    df_out.to_feather(output_filepath)
    print(f"Saved output to {output_filepath}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")
