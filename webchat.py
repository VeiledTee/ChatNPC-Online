import json
import os
from datetime import datetime
from typing import Any

import pinecone
from openai import OpenAI

import global_functions
from config import Config
from global_functions import (
    embed,
    extract_name,
    name_conversion,
    namespace_exist,
    prompt_engineer_from_template,
)
from keys import openAI_API, pinecone_API, pinecone_ENV
from retrieval import context_retrieval
from app import USERNAME
from flask import session


configuration = Config("config.json")
DATE_FORMAT = configuration.DATE_FORMAT
DEVICE = configuration.DEVICE
MODEL = configuration.MODEL
TOKENIZER = configuration.TOKENIZER

AUDIO: bool = True
# TEXT_MODEL: str = "gpt-3.5-turbo-0301"
TEXT_MODEL: str = "gpt-4-1106-preview"

# load api keys from openai and pinecone
client = OpenAI(api_key=openAI_API)
pinecone.init(
    api_key=pinecone_API,
    environment=pinecone_ENV,
)


def get_username():
    try:
        return session.get("username")
    except RuntimeError:
        return "penguins"


def get_information(character_name) -> None | tuple:
    """
    Return information from the character's data file
    :param character_name: name of character
    :return: None if character doesn't exist | profession and social status if they do exist
    """
    with open("Text Summaries/character_objects.json", "r") as character_file:
        data = json.load(character_file)

    for character in data["characters"]:
        if character["name"].strip() == character_name.strip():
            return character["profession"], character["social status"]

    return None  # character not found


def load_file_information(load_file: str) -> list[str]:
    """
    Loads data from a file into a list of strings for embedding.
    :param load_file: Path to a file containing the data we want to encrypt.
    :return: A list of all the sentences in the load_file.
    """
    with open(load_file, "r") as text_file:
        # list comprehension to clean and split the text file
        clean_string = [
            s.strip() + "."
            for line in text_file
            for s in line.strip().split(".")
            if s.strip()
        ]
    return clean_string


def run_query_and_generate_answer(
    query: str,
    receiver: str,
    context: list[str],
    index_name: str = "chatnpc-index",
    save: bool = True,
) -> str | tuple[str, int, int]:
    """
    Runs a query on a Pinecone index and generates an answer based on the response context.
    :param query: The query to be run on the index.
    :param receiver: The character being prompted
    :param context: The context required for the agent to answer the question correctly
    :param index_name: The name of the Pinecone index.
    :param save: A bool to save to a file is Ture and print out if False. Default: True
    :return: The generated answer based on the response context.
    """
    class_grammar_map: dict[str:str] = {
        "lower class": "poor",
        "middle class": "satisfactory",
        "high class": "formal",
    }

    history: list[dict] = []

    with open("Text Summaries/characters.json", "r") as f:
        character_names = json.load(f)

    profession, social_class = get_information(
        receiver
    )  # get the profession and social status of the character
    data_file: str = f"Text Summaries/Summaries/{character_names[receiver]}.txt"

    namespace: str = extract_name(data_file).lower()

    # Contradictory statement KB updates
    if query.lower() == "bye":
        return "Goodbye!"

    history = generate_conversation(data_file, history, True, query)

    # generate clean prompt and answer.
    clean_prompt = prompt_engineer_character_reply(
        query, class_grammar_map[social_class], context
    )
    save_prompt: str = clean_prompt.replace("\n", " ")

    generated_answer, prompt_tokens, reply_tokens = answer(
        clean_prompt, history, namespace
    )

    if save:
        # save results to file and return generated answer.
        with open("Text Summaries/prompt_response.txt", "a") as save_file:
            save_file.write("\n" + "=" * 120 + "\n")
            save_file.write(f"Prompt: {save_prompt}\n")
            save_file.write(f"To: {receiver}, a {profession}\n")
            clean_prompt = clean_prompt.replace("\n", " ")
            save_file.write(f"{clean_prompt}\n{generated_answer}")
    else:
        print(generated_answer)

    generate_conversation(
        f"Text Summaries/Summaries/{namespace.replace('-', '_')}.txt",
        chat_history=history,
        player=False,
        next_phrase=generated_answer,
    )

    update_history(
        namespace=namespace,
        info_file=data_file,
        prompt=query,
        response=generated_answer.split(": ")[-1],
        index=pinecone.Index(index_name),
    )

    return generated_answer, prompt_tokens, reply_tokens


def retrieve_context_list(
    namespace: str,
    query: str,
    impact_score: bool = True,
    n: int = 3,
    index_name: str = "chatnpc-index",
) -> list[str]:
    # embed query for processing
    embedded_query: list[float] = embed(query=query)
    # connect to index
    index = pinecone.Index(index_name)
    if impact_score:
        # retrieve memories using recency, poignancy, and relevance metrics
        return context_retrieval(
            namespace=namespace, query_embedding=embedded_query, n=n
        )
    else:
        # retrieve memories using ONLY cosine similarity
        responses = index.query(
            embedded_query,
            top_k=n,
            include_metadata=True,
            namespace=namespace,
            filter={
                "user": get_username(),
                "$or": [
                    {"type": {"$eq": "background"}},
                    {"type": {"$eq": "response"}},
                ],
            },
        )

        # Filter out responses containing the string "Player:"
        return [
            x["metadata"]["text"]
            for x in responses["matches"]
            if query not in x["metadata"]["text"]
        ]


def handle_contradiction(
    contradictory_index: int, namespace: str, index_name: str = "chatnpc-index"
) -> None:
    """
    This function deals with contradictory facts in the context and query.
    If the first fact (memory) is identified as false, remove it from the KB and replace with the second
    If the second fact (user input) is identified as false, ignore it
    If both facts are identified as true, store them as separate new facts in the KB
    If both facts are identified as false, remove the first from the KB and ignore the second
    :param contradictory_index: The index within the context list of the user's choice
    :param index_name: The name of the Pinecone index containing our vectors
    :param namespace: Namespace of the character user is speaking to
    :return: None, KB is updated within the function
    """
    index: pinecone.Index = pinecone.Index(index_name)

    # retrieve the s1 and s2 vectors for bookkeeping
    s1_vector = index.fetch(
        ids=["s1"],
        namespace=namespace,
    )
    s2_vector = index.fetch(ids=["s2"], namespace=namespace)

    # get current time in case we need to upsert
    cur_time = datetime.now()

    if contradictory_index == 0:  # premise already in KB
        index.delete(ids=["s1", "s2"], namespace=namespace)

    elif contradictory_index == 1:  # information presented by user
        # retrieve copy of s1 stored in main KB
        responses = index.query(
            s1_vector["vectors"]["s1"]["values"],
            top_k=1,
            include_metadata=True,
            namespace=namespace,
            filter={
                "user": get_username(),
                "$or": [
                    {"type": {"$eq": "background"}},
                    {"type": {"$eq": "response"}},
                ],
            },
        )
        # delete s1, s2, and s1 copy
        index.delete(
            ids=["s1", "s2", responses["matches"][0]["id"]], namespace=namespace
        )
        # upsert s2 (the true statement) into KB at the index of s1
        info_dict: dict = {
            "id": str(responses["matches"][0]["id"]),
            "metadata": {
                "text": s2_vector["vectors"]["s2"]["metadata"]["text"],
                "type": "response",
                "poignancy": find_importance(
                    namespace, s2_vector["vectors"]["s2"]["metadata"]["text"]
                ),
                "last_accessed": cur_time.strftime(DATE_FORMAT),
                "user": get_username(),
            },
            "values": s2_vector["vectors"]["s2"]["values"],
        }  # build dict for upserting
        index.upsert(vectors=[info_dict], namespace=namespace)

    elif contradictory_index == 2:  # both are correct, need to add premise 2
        index.delete(
            ids=["s1", "s2"], namespace=namespace
        )  # delete s1 and s2 for clean slate
        total_vectors: int = index.describe_index_stats()["namespaces"][namespace][
            "vector_count"
        ]  # get num of vectors existing in the namespace
        # upsert s2 (the true statement) into KB at the index of s1
        info_dict: dict = {
            "id": str(total_vectors),
            "metadata": {
                "text": s2_vector["vectors"]["s2"]["metadata"]["text"],
                "type": "response",
                "poignancy": find_importance(
                    namespace, s2_vector["vectors"]["s2"]["metadata"]["text"]
                ),
                "last_accessed": cur_time.strftime(DATE_FORMAT),
                "user": get_username(),
            },
            "values": s2_vector["vectors"]["s2"]["values"],
        }  # build dict for upserting
        index.upsert(vectors=[info_dict], namespace=namespace)

    elif contradictory_index == 3:  # remove original s1, s1, and s2
        # retrieve the s1 record
        s1_vector = index.fetch(ids=["s1"], namespace=namespace)
        # retrieve copy of s1 stored in main KB
        original_s1 = index.query(
            s1_vector["vectors"]["s1"]["values"],
            top_k=1,
            namespace=namespace,
            filter={
                "user": get_username(),
            },
        )
        # delete s1, s2, and s1 copy
        index.delete(
            ids=["s1", "s2", original_s1["matches"][0]["id"]], namespace=namespace
        )

        total_vectors: int = index.describe_index_stats()["namespaces"][namespace][
            "vector_count"
        ]  # get num of vectors existing in the namespace
        to_move_up = index.query(
            id=str(total_vectors),
            top_k=1,
            include_metadata=True,
            inclue_vectors=True,
            namespace=namespace,
            fileter={"user": get_username()},
        )  # get record with highest index value

        if to_move_up[
            "matches"
        ]:  # if we removed vector from the middle of the ID order
            index.delete(
                ids=[to_move_up["matches"][0]["id"]], namespace=namespace
            )  # remove duplicate
            info_dict: dict[str:Any] = {
                "id": original_s1["matches"][0]["id"],  # deleted vector ID
                "metadata": to_move_up["matches"][0][
                    "metadata"
                ],  # replacement metadata
                "values": to_move_up["matches"][0]["values"],  # replacement vector
            }  # build replacement dict
            index.upsert(
                vectors=[info_dict],
                namespace=namespace,
            )  # upsert replacement into where s1 originally stood


def generate_conversation(
    character_file: str, chat_history: list, player: bool, next_phrase: str
) -> list[dict]:
    """
    Generate a record of the conversation a user has had with the system for feeding into gpt 3.5 turbo
    :param character_file: The file associated with a character's background, not required unless
        first execution of function to 'set the stage'
    :param chat_history:
    :param player: is the player the one delivering the phrase
    :param next_phrase: the most recent phrase of the conversation
    :return: A list of dictionaries
    """
    if not chat_history:
        with open(character_file) as char_file:
            background: str = (
                f"You are {name_conversion(False, extract_name(character_file))}, "
                f"not an AI language model. Your background:"
            )
            for line in char_file.readlines():
                background += " " + line.strip()
        chat_history.append({"role": "system", "content": background})
    else:
        if player:
            chat_history.append({"role": "user", "content": next_phrase})
        else:
            chat_history.append({"role": "assistant", "content": next_phrase})
    return chat_history


def upload_background(character: str, index_name: str = "chatnpc-index") -> None:
    """
    Uploads the background of the character associated with the namespace
    :param character: the character we're working with
    :param index_name: the name of the pinecone index
    """
    if not pinecone.list_indexes():  # check if there are any indexes
        # create index if it doesn't exist
        pinecone.create_index("chatnpc-index", dimension=384, pods=1, pod_type="p2.x1")

    with open("Text Summaries/characters.json", "r") as character_info_file:
        character_names = json.load(character_info_file)

    if " " not in character:
        character = global_functions.name_conversion(
            to_snake_case=False, to_convert=character
        )

    data_file: str = f"Text Summaries/Summaries/{character_names[character]}.txt"
    # setting_file: str = "Text Summaries/Summaries/ashbourne.txt"
    namespace: str = extract_name(data_file).lower()
    # background has already been uploaded if namespace exists so can skip repeat uploads
    if namespace_exist(namespace):
        return None

    character_data: list[str] = load_file_information(data_file)
    # setting_data: list[str] = load_file_information(setting_file)
    data_facts: list[str] = []

    # data_facts.extend(character_data)
    # data_facts.extend(setting_data)

    for fact in character_data:
        data_facts.extend(fact_rephrase(fact, namespace, text_type="background"))
    # for fact in setting_data:
    #     data_facts.extend(fact_rephrase(fact, namespace, text_type='background'))

    index: pinecone.Index = pinecone.Index(index_name)

    total_vectors: int = 0
    data_vectors: list = []
    cur_time = datetime.now()

    for i, info in enumerate(data_facts):
        if i == 99:  # recommended batch limit of 100 vectors

            index.upsert(vectors=data_vectors, namespace=namespace)
            data_vectors = []
        info_dict: dict = {
            "id": str(total_vectors),
            "metadata": {
                "text": info,
                "type": "background",
                "poignancy": 10,
                "last_accessed": cur_time.strftime(DATE_FORMAT),
                "user": get_username(),
            },
            "values": embed(info),
        }
        data_vectors.append(info_dict)
        total_vectors += 1

    index.upsert(vectors=data_vectors, namespace=namespace)


def upload_contradiction(
    namespace: str, s1: str, s2: str, index_name: str = "chatnpc-index"
) -> None:
    """
    Sends text embedding vectors of two contradictory sentences into pinecone KB at indexes s1 and s2
    :param namespace: the pinecone namespace data is uploaded to
    :param s1: The contradictory sentence in the DB
    :param s2: The contradictory sentence presented by the user
    :param index_name: Name of the pinecone index to send data to
    """
    index: pinecone.Index = pinecone.Index(index_name)

    data_vectors: list = []
    cur_time = datetime.now()
    for i, info in enumerate([s1, s2]):
        info_dict: dict = {
            f"id": f"s{i + 1}",
            "metadata": {
                "text": info,
                "type": "response",
                "poignancy": 10,
                "last_accessed": cur_time.strftime(DATE_FORMAT),
                "user": get_username(),
            },
            "values": embed(info),
        }  # build dict for upserting
        data_vectors.append(info_dict)

    index.upsert(vectors=data_vectors, namespace=namespace)  # upsert all remaining data


def upload(
    namespace: str,
    data: list[str],
    index: pinecone.Index,
    text_type: str = "background",
) -> None:
    """
    Sends records to the namespace at the specified pinecone index
    :param namespace: the pinecone namespace data is uploaded to
    :param data: Data to be embedded and stored
    :param index: pinecone index to send data to
    :param text_type: The type of text we are embedding. Choose "background", "response", or "question".
        Default value: "background"
    """
    total_vectors: int = index.describe_index_stats()["namespaces"][namespace][
        "vector_count"
    ]  # get num of vectors existing in the namespace
    data_vectors = []
    # if a reply, break into facts
    if text_type == "response":
        data_facts: list[str] = []

        for fact in data:
            data_facts.extend(
                fact_rephrase(fact, namespace, text_type=text_type)
            )  # make facts out of statements

        data = data_facts

    # get current time memory is added
    cur_time = datetime.now()

    # add new data to database
    for i, info in enumerate(data):
        if i == 99:  # recommended batch limit of 100 vectors

            index.upsert(vectors=data_vectors, namespace=namespace)
            data_vectors = []
        info_dict: dict = {
            "id": str(total_vectors),
            "metadata": {
                "text": info,
                "type": text_type,
                "poignancy": find_importance(namespace, info),
                "last_accessed": cur_time.strftime(DATE_FORMAT),
                "user": get_username(),
            },
            "values": embed(info),
        }  # build dict for upserting
        data_vectors.append(info_dict)
        total_vectors += 1

    index.upsert(vectors=data_vectors, namespace=namespace)  # upsert all remaining data


def fact_rephrase(phrase: str, namespace: str, text_type: str) -> list[str]:
    """
    Given a sentence, break it up into individual facts.
    :param phrase: A phrase containing multiple facts to be distilled into separate ones
    :param namespace:
    :param text_type:
    :return: The split-up factual statements
    """
    msgs: list[dict] = []
    if text_type == "background":
        msgs.append(
            {
                "role": "system",
                "content": prompt_engineer_from_template(
                    template_file="Prompts/background_rephrase.txt",
                    data=[name_conversion(to_snake_case=False, to_convert=namespace)],
                ),
            }
        )

    # elif text_type == "response":
    #     msgs.append(
    #         {
    #             "role": "system",
    #             "content": prompt_engineer_from_template(
    #                 template_file="Prompts/response_rephrase.txt",
    #                 data=[name_conversion(to_snake_case=False, to_convert=namespace)],
    #             ),
    #         }
    #     )
    # elif text_type == "query":
    #     msgs.append(
    #         {
    #             "role": "system",
    #             "content": prompt_engineer_from_template(
    #                 template_file="Prompts/query_rephrase.txt",
    #                 data=[name_conversion(to_snake_case=False, to_convert=namespace)],
    #             ),
    #         }
    #     )
    prompt: str = f"Split this phrase into facts: {phrase}"
    msgs.append(
        {"role": "user", "content": prompt}
    )  # build current history of conversation for model

    res: Any = client.chat.completions.create(
        model=TEXT_MODEL, messages=msgs, temperature=0
    )  # conversation with LLM
    facts: str = str(res.choices[0].message.content).strip()  # get model response
    return [fact.strip() for fact in facts.split("\n")]


def find_importance(namespace: str, fact: str) -> int:
    character_name: str = name_conversion(to_snake_case=False, to_convert=namespace)
    with open(f"Text Summaries/Summaries/{namespace}.txt", "r") as character_file:
        character_info: str = character_file.read()

    prompt = prompt_engineer_from_template(
        template_file="Prompts/poignancy.txt",
        data=[character_name, character_info, character_name, fact],
    )  # get prompt for poignancy score

    example_output = 5
    fail_safe_output: int = 3  # used when the LLM can't assign a value

    prompt += f"ONLY include ONE integer value on the scale of 1 to 10 as the output.\n"
    prompt += "Example output integer:\n"
    prompt += str(example_output)

    res = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        top_p=1,
        max_tokens=1,
        stream=False,
        presence_penalty=0,
        frequency_penalty=0,
    )  # conversation with LLM
    clean_res: str = str(res.choices[0].message.content).strip()  # get model response

    try:
        return int(clean_res)
    except ValueError:  # if model replies with a string, assume failsafe output
        return fail_safe_output


def prompt_engineer_character_reply(
    prompt: str, grammar: str, context: list[str]
) -> str:
    """
    Given a base query and context, format it to be used as prompt
    :param prompt: The prompt query
    :param grammar: grammar the character uses based on social status
    :param context: The context to be used in the prompt
    :return: The formatted prompt
    """
    prompt_start: str = prompt_engineer_from_template(
        template_file="Prompts/character_reply.txt", data=[grammar]
    ).split("<1>")[0]
    with open("tried_prompts.txt", "a+") as prompt_file:
        if prompt_start + "\n" not in prompt_file.readlines():
            prompt_file.write(prompt_start + "\n")
    prompt_end: str = f"\n\nQuestion: {prompt}\nAnswer: "
    prompt_middle: str = ""
    # append contexts until hitting limit
    for c in context:
        print(f"Context: {c}")
        prompt_middle += f"\n{c}"
    return prompt_engineer_from_template(
        template_file="Prompts/character_reply.txt",
        data=[grammar, prompt_middle, prompt_end],
    )


def answer(
    prompt: str, chat_history: list[dict], namespace: str
) -> tuple[str, int, int]:
    """
    Using OpenAI API, respond to the provided prompt
    :param prompt: An engineered prompt to get the language model to respond to
    :param chat_history: the entire history of the conversation
    :param namespace: the namespace to save the records to
    :return: The completed prompt, the number of tokens used in the prompt, the number of tokens in the reply
    """
    with open("Text Summaries/text_to_speech_mapping.json", "r") as audio_model_info:
        model_pairings: dict = json.load(audio_model_info)
        cur_voice: str = model_pairings[namespace.replace("-", "_")]

    if not os.path.exists(
        f"static/audio/{get_username()}/{name_conversion(to_snake_case=False, to_convert=namespace)}"
    ):
        os.makedirs(
            f"static/audio/{get_username()}/{name_conversion(to_snake_case=False, to_convert=namespace)}"
        )

    msgs: list[dict] = chat_history
    msgs.append(
        {"role": "user", "content": prompt}
    )  # build current history of conversation for model
    # return "test"
    res: Any = client.chat.completions.create(
        model=TEXT_MODEL, messages=msgs, temperature=0
    )  # conversation with LLM
    clean_res: str = str(res.choices[0].message.content).strip()  # get model response
    if AUDIO:
        # generate audio file and save for output
        audio_reply = client.audio.speech.create(
            model="tts-1", voice=cur_voice, input=clean_res
        )
        # Add current time to the filename
        timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        filename = f"{namespace}_{timestamp}.mp3"

        audio_reply.stream_to_file(
            f"static/audio/{get_username()}/{name_conversion(to_snake_case=False, to_convert=namespace)}/{filename}"
        )
    return clean_res, int(res.usage.prompt_tokens), int(res.usage.completion_tokens)


def update_history(
    namespace: str,
    info_file: str,
    prompt: str,
    response: str,
    index: pinecone.Index,
) -> None:
    """
    Update the history of the current chat with new responses
    :param namespace: namespace of character we are talking to
    :param info_file: file where chat history is logged
    :param prompt: prompt user input
    :param response: response given by LLM
    :param index: The Pinecone index data will be saved to
    """
    upload(namespace, [prompt], index, text_type="query")  # upload prompt to pinecone
    upload(
        namespace, [response], index, text_type="response"
    )  # upload response to pinecone

    log_dir = "Text Summaries/Chat Logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Create the directory if it doesn't exist

    user_dir = os.path.join(log_dir, get_username())
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)  # Create the user directory if it doesn't exist

    info_file = os.path.join(user_dir, os.path.basename(info_file))  # Construct path

    with open(info_file, "a") as history_file:
        history_file.write(
            f"{get_username()}: {prompt}\n"
            f"{name_conversion(False, namespace).replace('-', ' ')}: {response}\n"
        )  # save chat logs


def are_contradiction(premise_a: str, premise_b: str) -> bool:
    # Tokenize the input
    tokenized_input = TOKENIZER(
        premise_a, premise_b, truncation=True, return_tensors="pt"
    )
    # Call the model to get the output
    output = MODEL(**tokenized_input)
    # Extract probabilities of each label
    prediction = output["logits"][0].softmax(dim=-1).tolist()
    # Define label names
    label_names = ["entailment", "neutral", "contradiction"]
    # Map probabilities to label names
    prediction = {
        name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)
    }
    # Calculate probability of non-contradiction
    prediction["non-contradiction"] = round(100 - prediction["contradiction"], 4)
    # Determine if contradiction probability is higher
    if prediction["contradiction"] >= prediction["non-contradiction"]:
        return True
    else:
        return False


if __name__ == "__main__":
    cur_time = datetime.now()
    index: pinecone.Index = pinecone.Index("chatnpc-index")
    s1_dict = {
        "id": "s1",
        "metadata": {
            "text": "This is a test boi",
            "type": "response",
            "poignancy": 6,
            "last_accessed": cur_time.strftime(DATE_FORMAT),
            "user": get_username(),
        },
        "values": embed("This is a test boi"),
    }
    s2_dict = {
        "id": "s2",
        "metadata": {
            "text": "Also a test boi",
            "type": "response",
            "poignancy": 5,
            "last_accessed": cur_time.strftime(DATE_FORMAT),
            "user": get_username(),
        },
        "values": embed(""),
    }
    # print(are_contradiction("blackfins only live in the east river", "blackfins only live in the west river"))
    # print(are_contradiction("blackfins live in the east river", "blackfins live in the west river"))
    handle_contradiction(contradictory_index=0, namespace="peter_satoru")
    print("0 works")
    index.upsert(vectors=[s1_dict, s2_dict], namespace="peter_satoru")
    handle_contradiction(contradictory_index=1, namespace="peter_satoru")
    print("1 works")
    index.upsert(vectors=[s1_dict, s2_dict], namespace="peter_satoru")
    handle_contradiction(contradictory_index=2, namespace="peter_satoru")
    print("2 works")
    index.upsert(vectors=[s1_dict, s2_dict], namespace="peter_satoru")
    handle_contradiction(contradictory_index=3, namespace="peter_satoru")
    print("3 works")
    index.upsert(vectors=[s1_dict, s2_dict], namespace="peter_satoru")
