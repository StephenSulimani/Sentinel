package math

import "math"

// NPV returns the net present value of cash flows received at the end of each
// period t=1..n, discounted at rate r (e.g. 0.10 for 10% per period).
func NPV(rate float64, flows []float64) float64 {
	if len(flows) == 0 {
		return 0
	}
	denom := 1.0 + rate
	var sum float64
	for i, cf := range flows {
		pow := math.Pow(denom, float64(i+1))
		sum += cf / pow
	}
	return sum
}
