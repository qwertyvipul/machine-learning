#install.packages("modeest")
library(modeest)

MyData <- c(1, 2, 4, 5, 8, 10, 2, 1, 33, 4, 5, 3, 5)
print(mlv(MyData, method = "mfv"))
barplot(MyData)