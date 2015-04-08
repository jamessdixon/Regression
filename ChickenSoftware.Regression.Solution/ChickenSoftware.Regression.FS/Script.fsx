
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

let x = Array.create numFeatures 1.
let x' = x |> Array.map(fun row -> (hi - lo) * rnd.NextDouble() + lo)

let xWts = Array.zip x' wts'
let xWts' = xWts |> Array.map(fun (x,wts) -> wts |> Array.sumBy(fun wt -> wt * x))

let y = Array.create numClasses 1.
let yWts = Array.zip y xWts'
let y' = yWts |> Array.map(fun (y,xwt) -> y + xwt)
 
let yBias = Array.zip y' biases'
let y'' = yBias |> Array.map(fun (y,bias) -> y + bias)

let maxVal = y'' |> Array.max

let y''' = y'' |> Array.map(fun y -> if y = maxVal then 1. else 0.)

let xy = Array.append x' y'''
let result = Array.create numRows xy

//78-80
//does not work -> same value in very slot of the array
//let x = Array.create numFeatures ((hi - lo) * rnd.NextDouble() + lo)
