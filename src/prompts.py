user_message_interpretations_gen = """Your task is to rewrite the question using a given word or phrase.

Examples:

Question:
Show titles of songs and names of singers.

Please rewrite using "stage name":
Give me titles of songs and stage names of singers.

Question:
Show the name of the conductor that has conducted the most number of orchestras.

Please rewrite using "director":
List the name of the director who has conducted the most number of orchestras.

Question:
Return the id of the document with the fewest paragraphs.

Please rewrite using "passages":
What is the id of the document with the fewest passages?

Please provide rewritten question for the following instance. Do not add any explanation or description, output only the rewritten question.

Question:
{}

Please rewrite using "{}":

"""

user_message_sql = """The task is to write SQL queries based on the provided questions in English. Questions can take the form of an instruction or command. Do not include any explanations, and do not select extra columns beyond those requested in the question.

Given the following SQLite database schema:

{}

Answer the following:
{}

"""

user_message_interpretations = """"Provide all possible interpretations for ambiguous questions in English. The question may be phrased as an instruction or command and will relate to the given database:

{}

Do not include explanations.

Question:
{}

"""


user_message_sql_ambig = """The task is to write SQL queries based on the provided questions in English. Questions can take the form of an instruction or command and can be ambiguous, meaning they can be interpreted in different ways. In such cases, write all possible SQL queries corresponding to different interpretations and separate each SQL query with an empty line. Do not include any explanations, and do not select extra columns beyond those requested in the question.

Given the following SQLite database schema:

{}

Answer the following:
{}

"""

user_message_sql_ambig_icl = """The task is to write SQL queries based on the provided questions in English. Questions can take the form of an instruction or command and can be ambiguous, meaning they can be interpreted in different ways. In such cases, write all possible SQL queries corresponding to different interpretations and separate each SQL query with an empty line. Do not include any explanations, and do not select extra columns beyond those requested in the question.

Some example databases, questions and corresponding SQL queries are provided based on similar problems:

EXAMPLES

Given the following SQLite database schema:

{}

Answer the following:
{}

"""

user_message_interpr_ambig = """You are tasked with analyzing questions and providing their possible interpretations. The questions are related to database queries and may be ambiguous or unambiguous.

Your task:
- List every distinct way the question could be understood
- Be thorough and consider all possible meanings
- Explore different ways the question could be interpreted
- Don't limit yourself to obvious interpretations

Important:
- List each interpretation on a separate line
- Do not include explanations or reasoning
- Focus on semantically different interpretations
- Be specific and precise in wording

Given the following database context:

{}

Provide interpretations for this question:
{}

"""

user_message_interpr_ambig_missing = """The task is to review the provided context, question, and existing interpretations, and determine if any additional interpretations are missing. If there are missing interpretations, list them on separate lines without explanations. If all interpretations have already been covered, simply state: "All possible interpretations are covered."

Given the following context:

{}

Question:

{}

Existing interpretations:

{}

Provide any missing interpretations or confirm that all possible interpretations are covered.
"""