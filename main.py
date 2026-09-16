from part_1a import pipelines as part1a_pipelines

def main() -> None:
    """Shows the pipeline menu and runs the user-selected pipeline."""
    print("-------------------------------------------------")
    print("This is the MAIR assignment from group B1 in 2026")
    print("-------------------------------------------------")
    print("Select a pipeline to run:")
    print("1. (part 1a) Training pipeline")
    print("2. (part 1a) Evaluation pipeline")
    print("3. (part 1a) Held-out test set pipeline")
    print("4. (part 1a) Difficult cases pipeline")
    print("5. (part 1a) Interactive classification pipeline")

    choice = input("Enter the number of the pipeline to run: ")
    if choice == "1":
        part1a_pipelines.run_training_pipeline()
    elif choice == "2":
        part1a_pipelines.run_evaluation_pipeline()
    elif choice == "3":
        part1a_pipelines.run_heldout_pipeline()
    elif choice == "4":
        part1a_pipelines.run_difficult_cases_pipeline()
    elif choice == "5":
        part1a_pipelines.run_interaction_pipeline()

if __name__ == "__main__":
    main()
