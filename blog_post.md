# "Defense wins championships": Coach's fact or fallacy?
## Analyzing whether this popular coach's axiom holds true in the context of NCAA men's basketball

"Offense wins games… Defense wins championships!" - an axiom that every athlete has heard from their coaches at every level of the game. As young athletes we don't question whether the phrase carries any truth or whether it's just meant to get us to stop thinking about our own points tally and start playing some defense, but as a data science student I am now in a position to do just that. In keeping with the season, as MarchMadness is upon us, I will explore the truthfulness of this quote in the context of NCAA basketball. In this article, I outline the process I used to clean and visualize the data from 10 NCAA Division I men's basketball seasons and how I used logistic regression to discern whether offense or defense have a stronger impact on winning championships.
The code from my analysis can be found within my github repository, here.

#### The Data
My analysis will feature this Kaggle dataset containing team statistics of each division I team for the 2013–2019 and 2021–2023 seasons (ignoring the 2020 COVID season). The main columns used in the analysis will be:
- TEAM
- YEAR
- W (wins)
- G (games)
- ADJOE (Adjusted Offensive Efficiency - An estimate of the offensive efficiency (points scored per 100 possessions) a team would have against the average Division I defense)
- ADJDE (Adjusted Defensive Efficiency - An estimate of the defensive efficiency (points allowed per 100 possessions) a team would have against the average Division I offense)

#### EDA
First I filtered the dataset to drop the teams that did not participate in the NCAA tournament as those teams would not have a high win count nor would they have had a chance to win the championship. Next, in order to avoid confusion, I created the column TEAM_YEAR by appending the contents of TEAM and YEAR. This also made sense in the context of college sports teams as the rosters are evolving frequently, so teams are often referred to by the season they played. I also had to create a dummy variable CHAMP (a binary column indicating 1 for yes and 0 for no) indicating whether each team won the championship or not.
Then, as we want to identify the keys to winning regular season games and the keys to winning championships, I decided to see how many of the winningest teams also won the championship that year to get a sense of how likely those keys are to coincide.

Figure 1: Bar plot displaying the number of wins of the 10 winningest teams in the database, coloured by whether they won the championship that season or notNext, to get a preliminary understanding of how good successful teams are on offense and defense, I created a scatter plot of offensive and defensive efficiency. I once again coloured the championship teams in the dataset in gold.
Figure 2: Scatter plot of adjusted offensive efficiency vs adjusted defensive efficiency coloured by championship status. The green line represents the average offensive efficiency, while the red line represents average defensive efficiencyWe can draw some very interesting conclusions from this plot, but first it is crucial to bear in mind that adjusted defensive efficiency is an estimate of the amount of points a team allows per 100 possessions, meaning that a lower ADJDE represents better defense. First of all, we see that all of the championship teams are above average in defensive efficiency and offensive efficiency, supporting the coach's axiom. Interestingly, the data point which is best (lowest) in defensive rating represents 2015 Kentucky - the team with the highest win count across the entire dataset, suggesting that perhaps, defense also wins games. However, this team also failed to win the championship despite having the best regular season of any team in the dataset.

#### The model
The EDA plots displayed early support for the coach's cliche. However, to test this support we will use logistic regression to model the relationship between offensive and defensive efficiency, and the response variable - whether or not the team won the championship. Then, by examining the regression coefficients we will be able to determine which is associated more strongly to winning a championship.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Prepare the features (X) and the target (y)
X = df[['ADJOE', 'ADJDE']]
y = df['CHAMP']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
logistic_model = LogisticRegression()

# Perform cross-validation
# cv specifies the number of folds; 5 is a common choice
scores = cross_val_score(logistic_model, X_train, y_train, cv=5)
```

Running 5-fold cross-validation we see that the training accuracy of the model is very high as we have a mean cross-validation accuracy of 0.98.
5-fold cross-validation resultsThe model also produced a test accuracy of 0.97 eliminating any worry of over-fitting and giving us confidence in the accuracy of our model. Let's go ahead and check the regression coefficients before we draw our final conclusions on the truthfulness of the coach's axiom.
Regression coefficientsThe regression coefficients tell us an increase in Adjusted Offensive Efficiency results in an increase in the log odds of being a champion while an increase in Adjusted Defensive Efficiency results in a decrease in the log odds of being a champion. To interpret these coefficients in terms of the odds of being a champion we must exponentiate them. 

```python
# To interpret the coefficients in terms of the odds of being a champion instead of the log odds, we must exponentiate the coefs

print(f"A 1 unit increase in Adjusted Offensive Efficiency results in a {np.exp(coefficients[0])} times increase in the odds of being a champion")
print(f"A 1 unit increase in Adjusted Defensive Efficiency results in a {np.exp(coefficients[1])} times decrease in the odds of being a champion")
```

We find that a 1 unit increase in Adjusted Offensive Efficiency results in a 1.32 times increase in the odds of being a champion. Meanwhile, a 1 unit increase in Adjusted Defensive Efficiency results in a 0.67 times decrease in the odds of being a champion.  Recall, it is key here to keep in mind that Adjusted Defensive Efficiency is an estimate of how many points a team allows over 100 possessions. Therefore, it makes sense that an increase in ADJDE, representing worse defense, would decrease the odds of winning a championship.

#### Conclusion
As defense is more strongly related to the odds of winning championships than offense, we can conclude that our coaches were right after all… Defense does win  championships.