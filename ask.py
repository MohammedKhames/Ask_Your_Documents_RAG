# Imports
from pathlib import Path

from rag_chain import build_rag_chain


# Run the command-line chat interface
def main():
    # Select a supported search method
    while True:
        search_type = input(
            "Search type (mmr/similarity): "
        ).strip().lower()

        if not search_type:
            search_type = "mmr"

        if search_type in ("mmr", "similarity"):
            break

        print("Please enter mmr or similarity.")

    # Prepare the RAG pipeline once
    try:
        chain = build_rag_chain(search_type)
    except FileNotFoundError as error:
        print(error)
        return

    print("\nReady! Type exit or quit to stop.")

    # Read questions and display answers
    while True:
        question = input("\nYou: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        result = chain.invoke(question)

        print("\nAnswer:")
        print(result["answer"])

        sources = sorted({
            Path(document.metadata["source"]).name
            for document in result["docs"]
        })

        print("Sources:", ", ".join(sources))


# Start the interface when this file is executed directly
if __name__ == "__main__":
    main()