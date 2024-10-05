# LLM metric extraction

Run some llms on manually annotated publications.



Scripts:
- pdf_parsing.py: With sciencbeam parser endpoint running localy this script parses pdfs and writes out the xml (to osm_output/pdf_texts).
- process_files.py: Aggregates metadata and xmls for both pdf sets, writing the xmls to tempdata/combined_pdfs
- run_llms.py:
  - Loads all xmls (removes the references section) and makes a train/test split (envirnment variable "VALIDATE" defines which is used).
  - "model" variable was editted manually to make use of different LLMs on openrouter. Models chosen must support tool use.
  - extract_using_model makes a few attempts at extracting the metrics. The openai tool calling interface is used.
  - The pydantic schema LLMExtractorMetrics is used to constrain model output to the desired schema.
  - A number of outputs are written to tempdata/llm_extractions in a standardized pattern.
- check_mid_run.py: Since the extraction across the traininng set can take a couple of hours this script can be used to check the progress of the extraction to see if it should be aborted for improvements in the prompts or an alternative model. The tsv file with "misses" or mismatches between manual and LLM outputs can be useful for assessing the quality of the extraction.
- plot_llm_performance.py: Compares oddpub, 4omini, and sonnet3.5 outputs to the manual annotations.

