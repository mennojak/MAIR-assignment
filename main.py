from part_1a import pipelines as pipelines_1a
from part_1b import pipelines as pipelines_1b

def main() -> None:
    """Shows the pipeline menu and runs the user-selected pipeline."""
    print("-------------------------------------------------")
    print("This is the MAIR assignment from group B1 in 2026")
    print("-------------------------------------------------")
    print("Select a pipeline to run:")
    print("Part 1a")
    print("1. (part 1a) Training pipeline")
    print("2. (part 1a) Evaluation pipeline")
    print("3. (part 1a) Held-out test set pipeline")
    print("4. (part 1a) Difficult cases pipeline")
    print("5. (part 1a) Interactive classification pipeline")
    print("Part 1b")
    print("6. (part 1b) Restaurant recommendation system pipeline")
    print("7. (part 1b) Reference dialogue tests pipeline")


    choice = input("Enter the number of the pipeline to run: ")
    print("-------------------------------------------------\n")
    if choice == "1":
        pipelines_1a.run_training_pipeline()
    elif choice == "2":
        pipelines_1a.run_evaluation_pipeline()
    elif choice == "3":
        pipelines_1a.run_held_out_pipeline()
    elif choice == "4":
        pipelines_1a.run_difficult_cases_pipeline()
    elif choice == "5":
        pipelines_1a.run_interaction_pipeline()
    elif choice == "6":
        pipelines_1b.run_interaction_pipeline()
    elif choice == "7":
        pipelines_1b.run_reference_dialog_tests_pipeline()

if __name__ == "__main__":
    main()
