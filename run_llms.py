from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import openai
import os
Path()
import json
import pickle
import logging
import traceback
# from fastapi import FastAPI, File, HTTPException, Query, UploadFile
# from llama_index.core import ChatPromptTemplate
# from llama_index.core.llms import LLM, ChatMessage
# from llama_index.llms.openai import OpenAI
# from llama_index.program.openai import OpenAIPydanticProgram
# from llama_index.core.prompts import PromptTemplate
# from pydantic import BaseModel, Field
# from llama_index.llms.openrouter import OpenRouter
from pydantic import ValidationError

import os
from datetime import datetime
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

from lxml import etree
from typing import Optional
import time

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
    scientific practices.
    Many unavailable identifiers (PMID, PMCID etc) can be found using pubmed: https://pubmed.ncbi.nlm.nih.gov/advanced/
    """

    model: str = Field(
        description="Exact verion of the llm model used to generate the data (not in publication itself but known by the model) e.g. GPT_4o_2024_08_06"
    )
    year: int = Field(
        description="Best attempt at extracting the year of the publication or use the int 9999",
    )
    journal: str = Field(description="The journal in which the paper was published")
    article_type: list[str] = Field(
        description="The type of article e.g. research article, review, erratum, meta-analysis etc.",
    )
    country: list[str] = Field(
        description="The countries of the affiliations of the authors",
    )
    institute: list[str] = Field(
        description="The institutes of the affiliations of the authors",
    )
    doi: str = Field(description="The DOI of the paper")
    pmid: int = Field(
        description="The PMID of the paper, use the integer 0 if one cannot be found"
    )
    pmcid: int = Field(
        description="The PMCID of the paper, use the integer 0 if one cannot be found"
    )
    title: str = Field(description="The title of the paper")
    authors: list[str] = Field(description="The authors of the paper")
    publisher: str = Field(description="The publisher of the paper")
    is_open_code: bool = Field(
        description="Whether there is evidence that the code used for analysis in the paper has been shared online",
    )
    code_sharing_statement: list[str] = Field(
        description="The statement in the paper that indicates whether the code used for analysis has been shared online",
    )
    is_open_data: bool = Field(
        description="Whether there is evidence that the data used for analysis in the paper has been shared online",
    )
    data_sharing_statement: list[str] = Field(
        description="The statement in the paper that indicates whether the data used for analysis has been shared online",
    )
    data_repository_url: str = Field(
        description="The URL of the repository where the data can be found"
    )
    dataset_unique_identifier: list[str] = Field(
        description="Any unique identifiers the dataset may have"
    )
    code_repository_url: str = Field(
        description="The URL of the repository where the code and data can be found"
    )
    has_coi_statement: bool = Field(
        description="Whether there is a conflict of interest statement in the paper",
    )
    coi_statement: list[str] = Field(
        description="The conflict of interest statement in the paper"
    )
    funder: list[str] = Field(
        description="The funders of the research, may contain multiple funders",
    )
    has_funding_statement: bool = Field(
        description="Whether there is a funding statement in the paper"
    )
    funding_statement: list[str] = Field(
        description="The funding statement in the paper"
    )
    has_registration_statement: bool = Field(
        description="Whether there is a registration statement in the paper",
    )
    registration_statement: list[str] = Field(
        description="The registration statement in the paper"
    )
    reasoning_steps: list[str] = Field(
        description="The reasoning steps used to extract the information from the paper",
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
                f"The llm model is {model}. The publication in xml follows below:\n"
                "------\n"
                f"{xml_content}\n"
                "------"
            ),
        }
    ]
def extract_using_model(xml_content: str, model: str) -> LLMExtractorMetrics:
    messages = get_initial_message(model, xml_content)
    for attempt in range(4):
        messages, result = attempt_extraction(messages, model)
        if result is not None:
            return result
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
        response_message = completion.choices[0].message
        tool_calls = response_message.tool_calls
        messages.append(response_message)
    except Exception as e:
        err = traceback.format_exc()
        messages.append({
            "role":"user",
            "content":f"{err} \n\n That last attempt resulted in the above error. Can we try again...",
        })
        return messages, None
    if not tool_calls:
        messages.append({
            "role":"user",
            "content":f"That doesn't have a tool call. Can we try again...",
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

    # model = "ai21/jamba-1-5-large"
    # model = "openai/gpt-3.5-turbo"
    # model = "openai/o1-mini-2024-09-12" doesn't have tools
    # model = "google/gemini-flash-1.5-exp" fails too often
    # model = "qwen/qwen-2.5-72b-instruct"
    # model = "openai/chatgpt-4o-latest"
    # model = "openai/gpt-4o-2024-08-06"
    # model = "anthropic/claude-3.5-sonnet"
    model = "openai/gpt-4o-mini-2024-07-18"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filepath = Path(f"tempdata/llm_extractions/{model.replace("/","-")}_{timestamp}.feather")
    print("output_filepath", output_filepath)
    if not output_filepath.parent.exists():
        output_filepath.parent.mkdir(parents=True)

    df = pd.read_feather("tempdata/combined_metadata.feather")
    # Perform the 90/10 split
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    with_xml = (
        train_df
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
            metrics = extract_using_model(xml_content, model).model_dump(mode="json")
            metrics['idx'] = idx
            outputs.append(metrics)
            output_filepath.with_suffix(".pkl").write_bytes(pickle.dumps(metrics))
        except Exception as e:
            err = traceback.format_exc()
            with open(output_filepath.with_suffix(".err"), "a") as log_file:
                log_file.write(f"Error processing row {idx}: {err}\n\n")
            logger.warning(f"Error processing row {idx}: {err[-200:]}...")
    df_llm = pd.DataFrame(outputs).set_index('idx')
    df_out = train_df.join(df_llm.rename(columns={col: f"llm_{col}" for col in df_llm.columns}))

    df_out.to_feather(output_filepath)
    logger.info(f"Saved output to {output_filepath}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")
