LessThan <- function(Value1, Value2){
  if(Value1 < Value2){
    result <- TRUE
  }else{
    result <- FALSE
  }
}

print(LessThan(1, 3))
print(LessThan(3, 1))