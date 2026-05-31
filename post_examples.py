import pandas as pd
import json


class PostExamples:
    def __init__(self, file_path="data/enriched_posts.json"):
        self.df = None
        self.unique_tags = None
        self.load_posts(file_path)

    # Reads the enriched JSON file, normalises it into a DataFrame, and derives the length category and unique tag list
    def load_posts(self, file_path):
        with open(file_path, encoding="utf-8") as f:
            posts = json.load(f)
            self.df = pd.json_normalize(posts)
            self.df['length'] = self.df['line_count'].apply(self.categorize_length)
            # collect unique tags
            all_tags = self.df['tags'].apply(lambda x: x).sum()
            self.unique_tags = list(set(all_tags))

    # Returns posts that match all three criteria (tag, language, length) as a list of dicts for use as few-shot examples
    def get_filtered_posts(self, length, language, tag):
        df_filtered = self.df[
            (self.df['tags'].apply(lambda tags: tag in tags)) &
            (self.df['language'] == language) &
            (self.df['length'] == length)
        ]
        return df_filtered.to_dict(orient='records')

    # Maps a raw line count to a Short / Medium / Long label used for filtering
    def categorize_length(self, line_count):
        if line_count < 5:
            return "Short"
        elif 5 <= line_count <= 10:
            return "Medium"
        else:
            return "Long"

    # Returns the deduplicated list of tags extracted across all loaded posts
    def get_tags(self):
        return self.unique_tags


if __name__ == "__main__":

    fs = PostExamples()
    
    # 1. See what tags actually exist
    print("Available tags:", fs.get_tags())
    
    # 2. See how many posts there are total
    print("Total posts:", len(fs.df))
    
    # 3. See what languages exist
    print("Languages:", fs.df['language'].unique())
    
    # 4. See the length distribution
    print("Lengths:", fs.df['length'].value_counts())
    
    # 5. Try the filter
    posts = fs.get_filtered_posts("Medium", "Hinglish", "Job Search")
    print("Matching posts:", posts)