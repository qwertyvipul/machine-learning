num <- as.integer(readline("Enter a number between 1 and 5: "))
result <- switch(
  num,
  "One",
  "2",
  "It's Three!",
  "Almost There!",
  "Done"
)
print(result)