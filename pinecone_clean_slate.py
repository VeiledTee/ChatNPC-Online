import json
from typing import List

import pinecone

from global_functions import (
    extract_name,
    name_conversion,
)
from keys import pinecone_API, pinecone_ENV
from webchat import get_information, load_file_information, upload_background


def delete_all_vectors(names_of_characters) -> None:
    for i in range(len(names_of_characters)):
        character: str = list(names.keys())[i]
        data: str = f"Text Summaries/Summaries/{names_of_characters[character].lower()}.txt"
        namespace: str = extract_name(data).lower()
        index.delete(deleteAll=True, namespace=namespace)


if __name__ == "__main__":
    pinecone.init(
        api_key=pinecone_API,
        environment=pinecone_ENV,
    )

    index = pinecone.Index("chatnpc-index")

    # Open file of characters and load its contents into a dictionary
    with open("Text Summaries/characters.json", "r") as f:
        names = json.load(f)

    delete_all_vectors(names)

    # loop through characters and store background in database
    for i in range(len(names)):
        # CHARACTER: str = 'Melinda Deek'
        CHARACTER: str = list(names.keys())[i]
        PROFESSION, SOCIAL_CLASS = get_information(CHARACTER)

        DATA_FILE: str = f"Text Summaries/Summaries/{names[CHARACTER].lower()}.txt"

        NAMESPACE: str = name_conversion(to_snake_case=True, to_convert=CHARACTER)
        char_info: List[str] = load_file_information(DATA_FILE)

        upload_background(CHARACTER)

        print(f"Background uploaded for {CHARACTER} --- Saved to namespace {NAMESPACE}")
