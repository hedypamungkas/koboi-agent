---
name: search-and-summarize
description: Research a topic by searching multiple sources and summarizing findings
license: MIT
allowed-tools: web_search web_fetch
---

# Search and Summarize

## Instructions

When this skill is activated, follow this structured approach:

1. **Plan**: Break the research question into sub-questions
2. **Search**: Use `web_search` to find information for each sub-question
3. **Fetch**: Use `web_fetch` to get detailed content from promising results
4. **Synthesize**: Combine findings into a coherent summary

## Output Format

Structure your response as:
- **Summary**: 2-3 sentence overview
- **Key Findings**: Bullet points with source references
- **Details**: Expanded explanation of each finding
- **Sources**: List of URLs consulted
