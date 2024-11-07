## Association Rules Mining with Apriori Algorithm

### Objective
To explore frequent country visitation patterns using association rule mining. By uncovering relationships between commonly visited countries, this analysis offers insights for potential joint tourism promotion strategies.

### Tools
- **Language**: R
- **Libraries**: arules for association rule mining

### Dataset Overview
A dataset of transactional country visitations, where each row represents a set of countries visited by a single traveler. The dataset includes 55 unique countries.

### Process
- **Data Preparation**: Loaded transactional data and inspected for frequently visited countries.
- **Apriori Algorithm**: Applied with minimum support of 0.2 and confidence of 0.8.
- **Parameter Tuning**: Adjusted support threshold iteratively to explore changes in rule generation.
- **Rule Inspection**: Sorted by support to identify top patterns, like the association between Cyprus and Greece.

### Modeling & Evaluation
- **Key Findings**:
  - *Cyprus* => *Greece*: High likelihood of visiting Greece after Cyprus.
  - Rules with high support indicate common travel patterns, suggesting actionable insights.
- **Parameter Tuning Visualization**:
![Support vs. Number of Rules Plot](images/Support%20vs.%20Number%20of%20Rules%20Plot.png)

### Business Implications
- **Tourism Strategy**: Joint tourism campaigns for Cyprus and Greece.
- **Marketing Insights**: Promote culturally similar regions together for enhanced visitor experience.

### Challenges and Improvements
- **Challenges**: Parameter tuning to balance rule quantity and relevance.
- **Future Improvements**: Explore other algorithms or finer threshold adjustments.

### Conclusion
This analysis highlights key visitation patterns, with actionable insights for tourism marketing in regions with frequent travel associations.
