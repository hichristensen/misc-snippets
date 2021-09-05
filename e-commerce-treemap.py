# libraries
import matplotlib.pyplot as plt
import squarify    # pip install squarify (algorithm for treemap)
import pandas as pd

# Create a data frame with fake data
df = pd.DataFrame({'commerce':[40.4,7.1,4.3,3.7,2.2,2.2,2.2,1.7,1.6,1.5],
                   'group':["Amazon", "Walmart", "eBay", "Apple", "Best Buy",
                            "Target", "Home Depot", "Kroger", "Costco", "Wayfair"] })

# plot it
squarify.plot(sizes=df['commerce'], label=df['group'], alpha=.8 )
plt.axis('off')
plt.show()
