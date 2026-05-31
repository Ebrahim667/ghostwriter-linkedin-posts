import json
from ai_engine import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


# Reads raw posts, enriches each with LLM-extracted metadata, unifies tags, and writes the result to the output file
def process_posts(raw_file_path, enriched_file_path=None):
    with open(raw_file_path, encoding='utf-8') as file:
        posts = json.load(file)
        enriched_posts = []
        for post in posts:
            # Fix broken surrogates but KEEP valid emojis
            clean_text = post['text'].encode('utf-16', 'surrogatepass').decode('utf-16')
            metadata = extract_metadata(clean_text)
            post_with_metadata = post | metadata
            enriched_posts.append(post_with_metadata)

    unified_tags = get_unified_tags(enriched_posts)
    for post in enriched_posts:
        current_tags = post['tags']
        new_tags = {unified_tags[tag] for tag in current_tags}
        post['tags'] = list(new_tags)

    with open(enriched_file_path, encoding='utf-8', mode="w") as outfile:
        json.dump(enriched_posts, outfile, indent=4)


# Calls the LLM to extract line_count, language, and tags from a single post and returns them as a dict
def extract_metadata(post):
    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
    1. Return a valid JSON. No preamble. 
    2. JSON object should have exactly three keys: line_count, language and tags. 
    3. tags is an array of text tags. Extract maximum two tags.
    4. Language should be English or Hinglish (Hinglish means hindi + english)
    
    Here is the actual post on which you need to perform this task:  
    {post}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"post": post})

    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse response.")
    return res


# Collects all tags across posts and asks the LLM to merge similar ones into a shorter, consistent set
def get_unified_tags(posts_with_metadata):
    unique_tags = set()
    # Loop through each post and extract the tags
    for post in posts_with_metadata:
        unique_tags.update(post['tags'])  # Add the tags to the set

    unique_tags_list = ','.join(unique_tags)

    template = '''I will give you a list of tags. You need to unify tags with the following requirements,
    1. Tags are unified and merged to create a shorter list. 
       Example 1: "Tech News", "AI Updates" can be merged into "Technology"
       Example 2: "Productivity Tips", "Time Management" can be merged into "Productivity"
       Example 3: "Founder Journey", "Startup Life" can be merged into "Entrepreneurship"
       Example 4: "Networking Tips", "Connections" can be merged into "Networking"
    2. Each tag should be follow title case convention. example: "Technology", "Productivity"
    3. Output should be a JSON object, No preamble
    4. Output should have mapping of original tag and the unified tag.
       For example: {{"Tech News": "Technology", "AI Updates": "Technology", "Productivity Tips": "Productivity"}}
    
    Here is the list of tags: 
    {tags}
    '''
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    response = chain.invoke(input={"tags": str(unique_tags_list)})
    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Context too big. Unable to parse response.")
    return res


if __name__ == "__main__":
    process_posts("data/source_posts.json", "data/enriched_posts.json")