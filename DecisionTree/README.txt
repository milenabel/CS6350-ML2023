Code files:

- carIDE.py using car datasets
- bankIDE.py using bank datasets

carIDE.py functionality:

- id3_algorithm(data, attributes, label, max_depth, heuristic): implements the ID3 algorithm:
- - majority_class(data, label): returns the class with the majority count in the data
- - information_gain(data, attribute, label): calculates the information gain for a given attribute
- - entropy(data): calculates the entropy of a dataset
- - gini_index(data, attribute, label): calculates the Gini index for a given attribute
- - gini(data): calculates the Gini index of a dataset
- - id3_recursive(data, attributes, label, max_depth, heuristic): handling different max_depth and heuristic values
- - majority_error(data, attribute, label): calculates the majority error for a given attribute

-  predict(tree, instance): makes predictions using the decision tree

- other code handles running the functions above and getting the needed results


bankIDE.py functionality:

- handle_missing_values(data, attributes): handles missing values by completing with majority value

- convert_numerical_to_binary(data, threshold_indices): converts numerical attributes to binary based on median

- id3_algorithm(data, attributes, label, max_depth, heuristic): implements the ID3 algorithm:
- - majority_class(data, label): returns the class with the majority count in the data
- - information_gain(data, attribute, label): calculates the information gain for a given attribute
- - entropy(data): calculates the entropy of a dataset
- - gini_index(data, attribute, label): calculates the Gini index for a given attribute
- - gini(data): calculates the Gini index of a dataset
- - id3_recursive(data, attributes, label, max_depth, heuristic): handling different max_depth and heuristic values
- - majority_error(data, attribute, label): calculates the majority error for a given attribute

-  predict(tree, instance): makes predictions using the decision tree

- other code handles running the functions above and getting the needed results
