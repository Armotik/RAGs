import pandas as pd
from dateparser.search import search_dates
from joblib import Parallel, delayed
from tqdm import tqdm


def extract_date(text: str, lang: str) -> tuple[list[str], str, str]:
    """
    Extract the date from the text.
    :param text: the text to extract the date from
    :param lang: the language of the text
    :return: the extracted date
    """

    results = search_dates(text, languages=[lang], settings={'STRICT_PARSING': True})
    if not results:
        return [], None, None

    iso_dates = []
    for _, date in results:
        try:
            iso_dates.append(date.date().isoformat())
        except AttributeError:
            continue

    # Trier et déterminer min/max
    iso_dates = sorted(set(iso_dates))
    earliest = min(iso_dates) if iso_dates else None
    latest = max(iso_dates) if iso_dates else None

    return iso_dates, earliest, latest


def enrich_meta_with_dates(row: pd.Series) -> pd.Series:
    """
    Enrich the metadata with dates.
    :param row: the row to enrich
    :return: the enriched row
    """
    iso_dates, earliest, latest = extract_date(row["text"], row["meta"]["lang"])
    meta = row["meta"].copy() if isinstance(row["meta"], dict) else {}

    meta["dates_iso"] = iso_dates
    meta["earliest_date"] = earliest
    meta["latest_date"] = latest
    return meta


def parallel_enrich_meta(df_chunk : pd.DataFrame, n_jobs=-1) -> pd.DataFrame:
    """
    Enrich the metadata with dates in with parallel processing.
    :param df_chunk: the dataframe chunk to enrich
    :param n_jobs: the number of jobs to run in parallel (-1 for all available)
    :return: the enriched dataframe chunk
    """

    rows = df_chunk.to_dict(orient="records")

    metas = Parallel(n_jobs=8)(
        delayed(enrich_meta_with_dates)(row) for row in tqdm(rows)
    )

    df_chunk["meta"] = metas
    return df_chunk
