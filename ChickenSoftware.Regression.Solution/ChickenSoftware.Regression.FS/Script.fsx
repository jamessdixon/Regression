
open System

let numFeatures = 4
let numClasses = 3
let numRows = 1000
let seed = 42

let rnd = new Random(seed)    
let hi = 10.0
let lo = -10.0

let wts = Array.create numFeatures (Array.create numClasses 1.)
let wts' = wts |> Array.map(fun row -> row |> Array.map(fun col -> (hi - lo) * rnd.NextDouble() + lo))

let biases = Array.create numClasses 1.
let biases' = biases |> Array.map(fun row -> (hi - lo) * rnd.NextDouble() + lo)






