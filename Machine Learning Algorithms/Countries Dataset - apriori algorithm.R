# Association Rules Mining with Apriori Algorithm
# Author: Alexandros Polyzoidis
# Date: April 1, 2024

# This script conducts itemset mining and association rule analysis on country visitation data.
# It applies the Apriori algorithm to uncover the relationships between different countries visited by travelers.

# Ensure the arules package is installed and loaded
if (!require(arules)) install.packages("arules")
library(arules)

# Set the correct path to the dataset
dataset_path <- "path/to/countries.csv" # Replace with the correct path

# Load the transactions from a CSV file
visits <- read.transactions(dataset_path,
                            format = "basket",
                            header=FALSE,
                            sep = ",",
                            rm.duplicates = FALSE)

# Dataset inspection
# Display a summary of the dataset to understand its composition
# Summary provides a quick overview of the transaction dataset including the number of transactions, items, and item frequency.
summary(visits)

# Identify the most frequently visited countries
frequent_countries <- itemFrequency(visits, type = "absolute")
top_country <- names(which.max(frequent_countries))

# Most frequently visited country : Greece
# Number of different countries visited : 55
# Item matrix density: 0.149
# Maximum number of countries visited by a traveler: 25
# Minimum number of countries visited by a traveler 3

# Apply the Apriori algorithm to the visits transaction dataset with a minimum support threshold of 0.2, 
# a minimum confidence threshold of 0.8 and a minimum of 2 items involved in a rule.
rules <- apriori(visits, parameter = list(supp=0.2, conf=0.8, minlen=2, target= "rules"))
summary(rules)

# The Apriori algorithm identified a total of 16 rules that satisfied the 
# minimum support threshold of 0.2 and a minimum confidence threshold of 0.8.
# The minimum rule length is 2 (which includes both the items on the left and right of the rule), and there are 8 rules of this length.
# The maximum rule length is 3, with 8 rules having this length as well.

# Number of identified rules: 16
# Number of rules with maximum number of items involved: 8
# Number of rules with minimum number of items involved: 8

# Sort the generated association rules in descending order by their support values.
# Support is a measure of how frequently the itemset appears in the dataset.
# Sorting by support helps prioritize rules that occur more often.
sorted_rules <- sort(rules, decreasing = TRUE, by="support")

# Inspect the top rules based on the sorted order. The 'inspect' function displays detailed information
# about each rule, including the items (or itemsets) on the left-hand side (LHS, antecedent) and
# right-hand side (RHS, consequent), along with the rule's support, confidence, and lift values.
# Here, 'n=16' specifies that details of the top 16 sorted rules are to be printed.
# Support indicates the proportion of transactions that contain the rule's items.
# Confidence measures the reliability of the inference made by the rule.
# Lift compares the rule's confidence with the expected confidence if the items were independent.
inspection <- inspect(sorted_rules,n=16)
print(inspection)

# {Belgium, Spain} => {France}
inspection[14,]

# {Hungary} => {Spain, France} does not exist, thus "N/A"

# {Belgium} => {Spain} does not exist, thus "N/A"

# {Cyprus} => {Greece}
inspection[6,]

# Setting initial parameters for the Apriori algorithm, including the minimum confidence level
# and the minimum number of items that must be involved in a rule. A sequence of support thresholds
# is defined, ranging from 0.125 to 0.25 with increments of 0.025.
min_confidence <- 0.8
min_items <- 2
support_thresholds <- seq(0.125, 0.25, by = 0.025)

# Initializing an empty vector to hold the number of rules generated for each support threshold.
num_rules_vector <- c()

# Iterating over each support threshold value to run the Apriori algorithm with the current
# support threshold, while keeping the confidence level and minimum item count constant.
# The number of generated rules is recorded for each threshold.
for (support in support_thresholds) {
  rules <- apriori(visits, parameter = list(support = support, confidence = min_confidence, minlen = min_items))
  num_rules <- length(rules)
  num_rules_vector <- c(num_rules_vector, num_rules)
  # Output the current support threshold and the corresponding number of generated rules
  # to the console for immediate observation.
  cat("Minimum Support:", support, "\t Number of Association Rules:", num_rules, "\n")
}


# Plotting the relationship between the minimum support thresholds and the number of generated
# association rules. This plot helps visually assess how increasing the support threshold
# impacts the number of rules, illustrating the trade-off between rule robustness and quantity.
plot(support_thresholds, num_rules_vector, type = "b", pch = 19, lwd = 2, col = "blue",
     main = "Number of Association Rules vs. Minimum Support Threshold",
     xlab = "Minimum Support Threshold", ylab = "Number of Rules")

# Explanations on the relationship between minimum support threshold and the number of rules

# The number of association rules declines as the minimum support threshold increases. 
# This decrease is most pronounced at lower thresholds, indicating many itemsets are frequent at these levels 
# but fail to appear as often at higher thresholds. 
# The trend suggests only a few itemsets are common enough to form rules when stricter support criteria are applied.

# Run the Apriori algorithm to find rules with Cyprus as the antecedent
# We're interested in rules where Cyprus leads to the purchase of other items (countries in this case).
Cyprus.lhs <- apriori(data = visits,
                      parameter = list(supp = 0.2, conf = 0.8, minlen = 2, target = "rules"),
                      appearance = list(lhs = c("Cyprus"), default = "rhs"))

# Inspect the rules to display those with Cyprus on the LHS.
# The inspect function will print the details of the rules meeting the specified criteria.
inspect(Cyprus.lhs)

# Thus, Italy and Greece are the consequents in the rules where Cyprus is the antecedent. 


# Interpretation in Traveler Patterns:
# The rule implies that when travelers visit Cyprus, there is a high likelihood (confidence = 1) that they will also visit Greece.
# This pattern suggests a strong association between visiting Cyprus and Greece among the travelers in the dataset.
# Travelers who visit Cyprus are very likely to include Greece in their travel itinerary based on the observed data.

# Implications for Cultural or Geographical Affinities:
# The high confidence level (1) indicates a strong relationship between visiting Cyprus and subsequently visiting Greece. 

# Practical Insights: Travel agencies or tourism boards could leverage this association to create joint travel packages or marketing strategies targeting travelers interested in experiencing both Cyprus and Greece.

# This association might be driven by various factors:
# Cultural Affinities: Cyprus and Greece share historical, cultural, and linguistic ties. Travelers interested in exploring cultural similarities may naturally include both destinations in their trips.
# Geographical Proximity: Cyprus and Greece are geographically close, making it convenient for travelers to explore both regions during a single trip.
# Similar Touristic Appeal: Both countries are known for their Mediterranean charm, historical sites, and scenic landscapes. Travelers with an interest in such attractions may choose to visit both destinations.
