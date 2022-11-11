## Abstract

The vast majority of all digital "search" that the average person performs is 
exact matching.

This is sub-optimal, as concept-based similarity and matching is much closer to
how humans think than the purely text similarities that are the standard.

There is significant scope to improve the user experience by incorporating 
concept and semantic search, especially on small sized corpora.


## Objective

This project aims to implement a framework which could then be deployed on 
arbitrary documents/FAQ's with minimal overhead.

An abstraction layer is to be provided to enable users to query a document/corpus, and get 
relevant answers.

What is *not* in scope:
- Any medium other than text (i.e audio, images, videos, etc.)
- Generative models—stakeholders need to have confidence that all outputs
from the model(s) originate in the source documents *only*, and the worst case scenario
is selection of the wrong section/answer.


## Methodologies

There are at present two main forms in which textual knowledge is stored:
- Structured Question/Answer pairs (FAQ's)
- "Unstructured" documents (which have some structure in the form of headings/sections)

Both classical and modern deep learning methods are to be applied, with the method 
selected based on dataset size and whether the data is structured/unstructured.


## Possible Outcomes

- Improve usability/accessibility of documentation/knowledge articles to non-experts
- Reduce the manual effort involved in creating and maintaining general FAQ's


## Applicability in the Real World

This framework would be of great relevance within organizations with an internal 
knowledge base which is currently indexed and curated by hand.

The ability to query an unstructured corpus in natural language and extract the 
key section/paragraph from it, or more optimistically, the *exact* answer, 
would be invaluable.
