import pandas as pd

# Define the list of tasks to process
tasks = ["aclue", "arc_c", "cmmlu", "hotpot_qa", "math", "mmlu", "squad"]

for task in tasks:
    train_file = f"{task}_train.csv"
    test_file = f"{task}_test_pred.csv"
    try:
        # Read the training CSV for the task
        train_df = pd.read_csv(train_file)
    except FileNotFoundError:
        print(f"Warning: Training file {train_file} not found. Skipping task '{task}'.")
        continue
    except Exception as e:
        print(f"Warning: Could not read {train_file} due to error: {e}. Skipping '{task}'.")
        continue

    # Determine which columns are model accuracy columns
    model_columns = []
    for col in train_df.columns:
        col_lower = col.lower()
        # Skip non-model columns by name convention or data type
        if col_lower in ("id", "question", "prompt"):
            continue
        # Only consider numeric columns as model accuracies
        col_data = train_df[col]
        if col_data.dtype in [float, int] or str(col_data.dtype).startswith(("float", "int")):
            # Verify values range between 0 and 1 (likely accuracy indicators)
            valid_vals = col_data.dropna()
            if len(valid_vals) == 0:
                # If column has no data, skip it
                continue
            # If any value is outside [0,1], this is probably not an accuracy column
            if valid_vals.min() < 0 or valid_vals.max() > 1:
                continue
            model_columns.append(col)
    if not model_columns:
        print(f"Warning: No model columns found in {train_file}. Skipping task '{task}'.")
        continue

    # Calculate average accuracy (bid) for each model
    bids = {}
    for model in model_columns:
        # Compute mean accuracy for the model (skip NaN values if any)
        mean_acc = train_df[model].mean(skipna=True)
        # Only consider the model if mean is a valid number
        if pd.isna(mean_acc):
            continue
        bids[model] = mean_acc

    if not bids:
        print(f"Warning: No valid accuracy data in {train_file}. Skipping task '{task}'.")
        continue

    # Determine the winner and second-highest bid
    # Sort models by their bid value (descending)
    sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
    best_model, best_bid = sorted_bids[0]         # highest bid and model name
    if len(sorted_bids) > 1:
        second_price_value = sorted_bids[1][1]    # second-highest bid value
    else:
        second_price_value = best_bid

    # (For info) Print the chosen model and second price value
    print(f"Task '{task}': Winning model = {best_model} (bid={best_bid:.4f}), second price = {second_price_value:.4f}")

    # Read the test CSV for the task
    try:
        test_df = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Warning: Test file {test_file} not found. Skipping task '{task}'.")
        continue
    except Exception as e:
        print(f"Warning: Could not read {test_file} due to error: {e}. Skipping '{task}'.")
        continue

    # If 'pred' column doesn't exist, create it. If it exists (even filled), we'll overwrite it.
    if "pred" not in test_df.columns:
        test_df["pred"] = ""

    # Fill the 'pred' column with the winning model name for all rows
    test_df["pred"] = best_model

    # Save the updated test predictions back to CSV (overwriting the file)
    try:
        test_df.to_csv(test_file, index=False)
    except Exception as e:
        print(f"Warning: Could not write to {test_file}: {e}.")
