'''
The idea of regression - take continuous data and figure out the best fit line
for that data and then perform predictions.

Just like the equation [y = mx + c]
'''

import pandas as pd
import quandl

#DATAFRAME
'''
Visit quandl.com and search for the datasets
'''
df = quandl.get('WIKI/GOOGL')

print(df.head()) # To see what it is we are working with

'''
Now we do not want un-necessary data - so we will trim the data
and work only on the relevant data
'''

