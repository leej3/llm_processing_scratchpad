future::plan(future::multisession)
oddpub::pdf_convert("tempdata/combined_pdfs", "oddpub_output")
PDF_text_sentences <- oddpub::pdf_load("oddpub_output/")
cleaned<-within(PDF_text_sentences,rm("https_colon_slash_slash_doi.org_slash_10.7554_slash_eLife.92805.1.txt"))

open_data_results <- oddpub::open_data_search(cleaned)
cleaned.