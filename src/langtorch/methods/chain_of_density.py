from .. import TextTensor, TextModule, ActivationGPT, Text


class CoDModule(TextModule):
    def __init__(self, iterations=5):
        # Base CoD prompt template
        self.template = """
        Article: {*}

        You will generate increasingly concise, entity-dense summaries of the above Article.
        Repeat the following 2 steps {} times.

        Step 1. Identify 1-3 informative Entities ("; " delimited) from the Article which are missing from the previously generated summary.
        Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.

        A Missing Entity is:
        - Relevant: to the main story.
        - Specific: descriptive yet concise (5 words or fewer).
        - Novel: not in the previous summary.
        - Faithful: present in the Article.
        - Anywhere: located anywhere in the Article.

        Guidelines:
        - The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
        - Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
        - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
        - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
        - Missing entities can appear anywhere in the new summary.
        - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

        Remember, use the exact same number of words for each summary.
        """.format(iterations)

        super().__init__(self.template,
                         ActivationGPT(system_message="You are a summarizer.", T=0.9, model="gpt-3.5-turbo"))

    def forward(self, article: TextTensor) -> TextTensor:
        # Process the article with the CoD prompt
        result = self.activation(self.prompt * article)
        return result

# Example usage
# article = TextTensor(
#     [Text(("Article", "This is an example article content that will be summarized using the CoD technique."))])
# CoD = CoDModule()
# result = cod_module(article)
#
# # Extracting results from the TextTensor and formatting it for output
# output = [{"Missing_Entities": entry.content[0].get("Missing_Entities", ""),
#            "Denser_Summary": entry.content[0].get("Denser_Summary", "")} for entry in result.content]
# print(output)
