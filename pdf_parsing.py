from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from osm.pipeline.parsers import ScienceBeamParser


def parse_pdfs_to_xml(unprocessed):
    for index, row in unprocessed.iterrows():
        fname = Path(row.filename)
        data = fname.read_bytes()
        try:
            parsed= ScienceBeamParser()._run(data)
            (Path("osm_output/pdf_texts") / f"{fname.stem}.xml").write_bytes(parsed)
        except Exception as e:
            breakpoint()
            pass
def move_xmls_to_doi_based_name(df_in):
    for index, row in df_in.iterrows():
        doi = row.doi
        if doi:
            doi_path = Path("osm_output/pdf_texts") / (doi + ".xml")
            if doi_path.exists():
                doi_path.rename(Path("osm_output/pdf_texts") / (row.filename.stem + ".xml"))

if __name__ == "__main__":
    df = pd.read_feather("tempdata/combined_pdfs/combined_metadata.feather")
    # unprocessed = df.query("year.notnull()")
    # parsed_pdfs = parse_pdfs_to_xml(unprocessed)
    nimh = df.query("year.isnull()")
    parsed_pdfs = parse_pdfs_to_xml(nimh)

    # train_df, test_df = train_test_split(unprocessed, test_size=0.1, random_state=42)

