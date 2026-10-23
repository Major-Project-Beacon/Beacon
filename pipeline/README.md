# BEACON Fact Extraction Pipeline

## Overview

The BEACON Fact Extraction Pipeline processes meeting transcripts and extracts structured, evidence-backed information from conversations.

It converts unstructured meeting dialogue into a machine-readable JSON output containing key facts such as discussion points, decisions, action items, commitments, people mentioned, and relevant dates or deadlines.

The extracted information can be used by downstream BEACON components for meeting understanding, summarization, compliance analysis, and knowledge organization.

## Objectives

- Extract important information from meeting transcripts.
- Preserve evidence from the original conversation for extracted facts.
- Identify decisions, tasks, commitments, and other meeting information.
- Produce structured JSON that downstream modules can consume.
- Reduce unsupported or inferred claims by grounding extraction in the transcript.
- Support integration with the wider BEACON pipeline.

## Pipeline Workflow

The pipeline follows these general stages:

1. **Transcript input**  
   Load the meeting transcript supplied to the extraction program.

2. **Prompt preparation**  
   Prepare instructions that guide the model to extract relevant facts using the transcript as the source.

3. **LLM-based fact extraction**  
   Submit the transcript and extraction instructions to the configured Gemini model.

4. **Response processing**  
   Process the model response and obtain the structured extraction result.

5. **Validation**  
   Check that the response follows the expected JSON structure and required fields.

6. **Output generation**  
   Save the extracted information as a JSON file for inspection or use by subsequent BEACON modules.

7. **Downstream integration**  
   Make the validated facts available to other pipeline components.

> The exact input paths, validation rules, and output schema are defined by the implementation. Refer to the extraction script and its configuration for the authoritative details.

## Information Extracted

Depending on the transcript and the extraction schema, the output may include:

| Information category | Description |
|---|---|
| Key points | Important discussion topics and meeting highlights |
| Decisions | Decisions or proposed decisions, including their status when supported |
| Action items | Tasks, assigned owners, and deadlines when stated |
| Commitments | Promises or responsibilities agreed upon by participants |
| People mentioned | Participants and other individuals referenced |
| Dates and deadlines | Dates, milestones, and time-related obligations |
| Supporting evidence | Transcript excerpts that support extracted information |

The pipeline should not treat a suggestion as an approved decision, assign an owner without evidence, or invent a deadline when the transcript does not provide one.

## Technology Stack

- **Python** – pipeline implementation
- **Google Gemini API** – LLM-based fact extraction
- **JSON** – structured extraction output
- **Git and GitHub** – source control and team collaboration

The exact Python packages and model configuration are specified in the project’s dependency and extraction files.

## Repository Structure

The fact-extraction branch contains the pipeline module. A simplified structure is shown below:

```text
pipeline/
├── README.md
├── .gitignore
├── requirements.txt
├── <fact-extraction-script>.py
├── <supporting-modules-or-files>
└── <generated-output-files, if tracked>
```

Replace the placeholder filenames above with the actual filenames present in the repository. Generated meeting outputs and private configuration files should generally not be committed unless the project specifically requires them.

## Requirements

Before running the pipeline, ensure that you have:

- Python installed at a version supported by the project dependencies.
- Access to the BEACON fact-extraction source code.
- A valid Gemini API key.
- A meeting transcript in the format expected by the extraction script.
- The required Python packages installed.

## Setup

### 1. Clone the repository

Clone the BEACON repository and switch to the fact-extraction branch:

```bash
git clone <BEACON_REPOSITORY_URL>
cd <BEACON_REPOSITORY_FOLDER>
git switch fact-extraction-module
```

If you already have the repository locally, navigate to its root and switch to the branch.

### 2. Navigate to the pipeline directory

```bash
cd pipeline
```

### 3. Install dependencies

Install the packages listed in the pipeline's dependency file:

```bash
python -m pip install -r requirements.txt
```

If the project uses a different dependency file or environment setup, follow that configuration instead.

### 4. Configure the API key

Provide your Gemini API key through the configuration mechanism expected by the extraction script.

If the script reads an environment variable, configure it in your local environment or in a local `.env` file. For example, if the script expects `GEMINI_API_KEY`:

**PowerShell:**

```powershell
$env:GEMINI_API_KEY = "YOUR_API_KEY"
```

Confirm the actual variable name in the script before using this example. Never commit real API keys, credentials, or private `.env` files to GitHub.

## Running the Pipeline

Run the fact-extraction script from the directory and with the arguments expected by its implementation.

Example command pattern:

```bash
python <fact-extraction-script>.py
```

Replace the placeholder with the actual script filename. If the script accepts a transcript path, model setting, or output path as command-line arguments, provide those according to its help text or source code.

### Input

The pipeline expects a meeting transcript. The transcript may contain speaker labels and timestamps if they are available in the source.

The input path and supported transcript format are determined by the script configuration.

### Output

The pipeline produces structured JSON containing the extracted meeting information.

The output location and filename are determined by the script configuration. Inspect the generated JSON to confirm:

- It is valid JSON.
- Its fields match the expected schema.
- Extracted facts are supported by the transcript.
- Evidence excerpts accurately reflect the source.
- Unknown or unstated information is not presented as certain.

## Reliability and Evidence Grounding

Fact extraction from meeting transcripts can be affected by transcription errors, incomplete speaker labels, ambiguous statements, or unclear decisions.

The pipeline should therefore:

- Use the transcript as the source of truth for extraction.
- Preserve uncertainty when a statement is ambiguous.
- Avoid inventing names, dates, owners, or decision statuses.
- Distinguish proposals from confirmed decisions.
- Include supporting transcript evidence where the output schema requires it.
- Validate the generated JSON before passing it to another module.

A valid JSON response does not automatically mean that every extracted fact is correct. Outputs should be reviewed against the transcript when accuracy is important.

## Error Handling and Troubleshooting

| Issue | Suggested check |
|---|---|
| API authentication failure | Verify that the API key is configured correctly and has access to the selected model |
| Missing Python package | Install the dependencies listed in `requirements.txt` |
| Transcript file not found | Check the configured input path and filename |
| Invalid JSON output | Inspect the model response and the script's parsing and validation logic |
| Empty or incomplete extraction | Check transcript quality, prompt instructions, model response, and any length limits |
| API rate limit or temporary failure | Check the API response and the script's retry handling |
| Incorrect or unsupported facts | Compare the extracted facts and evidence with the original transcript |

## GitHub Collaboration

The pipeline is maintained in the BEACON repository on the `fact-extraction-module` branch.

To update the pipeline:

1. Make changes to the pipeline files.
2. Review the changes and confirm that no credentials or private data are included.
3. Commit the intended pipeline changes.
4. Push the commit to the fact-extraction branch.

Example:

```bash
git switch fact-extraction-module
git pull origin fact-extraction-module
git add pipeline/
git commit -m "Update BEACON fact extraction pipeline"
git push origin fact-extraction-module
```

Git tracks individual files inside the folder. Adding, modifying, or removing files in `pipeline/` will be reflected in the commit.

## Security

- Keep API keys and credentials outside tracked source files.
- Do not commit `.env` files containing secrets.
- Use `.gitignore` to exclude local configuration and generated files that should not be tracked.
- If a secret is accidentally pushed, revoke or rotate it. Removing it in a later commit does not erase it from the repository's history.

## Current Scope

This module focuses on extracting structured facts from meeting transcripts. It is one component of BEACON and is intended to provide machine-readable information for subsequent processing.

The exact extraction categories, schema, model configuration, and integration behavior may evolve as the BEACON pipeline develops.

## Contributors

Maintained by the BEACON project team.

For project-specific questions, consult the source code, dependency file, and documentation in this branch.
