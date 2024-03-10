import json
import re

import torch
from sentence_transformers import SentenceTransformer
import pinecone
import psutil


def embed(query: str) -> list[float]:
    """
    Take a sentence of text and return the 384-dimension embedding
    :param query: The sentence to be embedded
    :return: Embedding representation of the sentence
    """
    # create SentenceTransformer model and embed query
    model = SentenceTransformer("all-MiniLM-L6-v2")  # fast and good, 384
    # model = SentenceTransformer("all-mpnet-base-v2")  # slow and great, 768
    device = "cuda" if torch.cuda.is_available() else "cpu"  # gpu check
    model = model.to(device)
    return model.encode(query).tolist()


def extract_name(file_name: str) -> str:
    """
    Extracts the name of a character from their descriptions file name
    :param file_name: the file containing the character's description
    :return: the character's name, separated by a hyphen
    """
    # split the string into a list of parts, using "/" as the delimiter
    parts = file_name.split("/")
    # take the last part of the list (i.e. "john_pebble.txt")
    filename = parts[-1]
    # split the filename into a list of parts, using "_" as the delimiter
    name_parts = filename.split("_")
    # join the name parts together with a dash ("-"), and remove the ".txt" extension
    name = "_".join(name_parts)
    return name[:-4]


def name_conversion(to_snake: bool, to_convert: str) -> str:
    """
    Convert a namespace to character name (not snake) or character name to namespace (snake)
    :param to_snake: Do you convert to namespace or not
    :param to_convert: String to convert
    :return: Converted string
    """
    if to_snake:
        text = to_convert.lower().split(" ")
        converted: str = text[0]
        for i, t in enumerate(text):
            if i == 0:
                pass
            else:
                converted += f"_{t}"
        return converted
    else:
        text = to_convert.split("_")
        converted: str = text[0].capitalize()
        for i, t in enumerate(text):
            if i == 0:
                pass
            else:
                converted += f" {t.capitalize()}"
        converted = re.sub("(-)\s*([a-zA-Z])", lambda p: p.group(0).upper(), converted)
        return converted.replace("_", " ")


def namespace_exist(namespace: str) -> bool:
    """
    Check if a namespace exists in Pinecone index
    :param namespace: the namespace in question
    :return: boolean showing if the namespace exists or not
    """
    index = pinecone.Index("chatnpc-index")  # get index
    responses = index.query(
        embed(" "),
        top_k=1,
        include_metadata=True,
        namespace=namespace,
        filter={
            "$or": [
                {"type": {"$eq": "background"}},
                {"type": {"$eq": "response"}},
            ]
        },
    )  # query index
    return (
        responses["matches"] != []
    )  # if matches comes back empty namespace doesn't exist


def prompt_engineer_from_template(template_file: str, data: list[str]) -> str:
    """
    Takes the template of a prompt to be given to OpenAI GPT models and fills the blank spaces with provided information
    :param template_file: Path to the prompt template we need
    :param data: A list of strings containing the data to inject into the prompt
    :return: The template filled with the correct information
    """
    with open(template_file, "r") as file:
        prompt = file.read()
    for index, information in enumerate(data):
        prompt = prompt.replace(f"<{index}>", information)
    return prompt


def get_network_usage():
    """
    Gets the number of bytes sent and received using psutil
    :return: bytes sent, bytes received
    """
    net_io = psutil.net_io_counters()
    return net_io.bytes_sent, net_io.bytes_recv


def delete_all_vectors(index_name: str = "chatnpc-index") -> None:
    """
    Deletes all vectors and namespaces in the pinecone database
    :param index_name: name of the index data is stored in
    :return: None
    """
    index: pinecone.Index = pinecone.Index(index_name)

    with open("Text Summaries/characters.json", "r") as f:
        names = json.load(f)

    for i in range(len(names)):
        character: str = list(names.keys())[i]
        print(character)
        data: str = f"Text Summaries/Summaries/{names[character].lower()}.txt"
        namespace: str = extract_name(data).lower()
        index.delete(deleteAll=True, namespace=namespace)


def delete_specific_vectors(
    character_name: str, index_name: str = "chatnpc-index"
) -> None:
    """
    Deletes all vectors in a specific namespace
    :param character_name: character's name who's memory needs to be wiped (namespace or full name)
    :param index_name: Name of the pinecone index everything is stored in
    :return: None
    """
    index: pinecone.Index = pinecone.Index(index_name)
    if "_" not in character_name:
        namespace: str = name_conversion(to_snake=True, to_convert=character_name)
    index.delete(deleteAll=True, namespace=namespace)


if __name__ == "__main__":
    with open("keys.txt", "r") as key_file:
        api_keys = [key.strip() for key in key_file.readlines()]
        pinecone.init(
            api_key=api_keys[1],
            environment=api_keys[2],
        )

    delete_specific_vectors("Sarah Ratengen")
