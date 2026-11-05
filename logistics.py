
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("03-decision-tree_1.csv", na_values="?")

# Convert target to categorical
df["Direct"] = df["Direct"].astype("category")

# Exclude Route and Draught
x = ["shiptype", "Length", "Breadth", "SOG",
     "TotalDistance", "Duration"]
y = df["Direct"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[x], y, test_size=0.2, random_state=1
)

# Initial decision tree
dt = DecisionTreeClassifier(random_state=1)
dt.fit(X_train, y_train)

predictions = dt.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Initial accuracy:", accuracy)
print("Depth:", dt.get_depth())
print("Nodes:", dt.tree_.node_count)


# Test different maximum depths
for depth in range(1, 16):
    model = DecisionTreeClassifier(
        random_state=1, max_depth=depth
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(
        depth,
        accuracy_score(y_test, predictions)
    )

# Test different minimum leaf sizes
for leaf in [1, 2, 5, 10, 20, 30, 50, 75, 100]:
    model = DecisionTreeClassifier(
        random_state=1, min_samples_leaf=leaf
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(
        leaf,
        accuracy_score(y_test, predictions)
    )

# Cost-complexity pruning
full_tree = DecisionTreeClassifier(random_state=1)
full_tree.fit(X_train, y_train)

path = full_tree.cost_complexity_pruning_path(
    X_train, y_train
)

alphas = path["ccp_alphas"]

best_accuracy = 0
best_alpha = None
best_tree = None

for alpha in alphas:
    model = DecisionTreeClassifier(
        random_state=1, ccp_alpha=alpha
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_alpha = alpha
        best_tree = model

print("Best alpha:", best_alpha)
print("Best accuracy:", best_accuracy)
print("Tree depth:", best_tree.get_depth())
print("Number of nodes:", best_tree.tree_.node_count)


# Cost-complexity pruning
full_tree = DecisionTreeClassifier(random_state=1)
full_tree.fit(X_train, y_train)

path = full_tree.cost_complexity_pruning_path(
    X_train, y_train
)

alphas = path["ccp_alphas"]

best_accuracy = 0
best_alpha = None
best_tree = None

for alpha in alphas:
    model = DecisionTreeClassifier(
        random_state=1, ccp_alpha=alpha
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_alpha = alpha
        best_tree = model

print("Best alpha:", best_alpha)
print("Best accuracy:", best_accuracy)
print("Tree depth:", best_tree.get_depth())
print("Number of nodes:", best_tree.tree_.node_count)