package math

import (
	"math"
	"testing"
)

func TestNPV_Empty(t *testing.T) {
	if got := NPV(0.1, nil); got != 0 {
		t.Fatalf("NPV(nil) = %v, want 0", got)
	}
	if got := NPV(0.1, []float64{}); got != 0 {
		t.Fatalf("NPV(empty) = %v, want 0", got)
	}
}

func TestNPV_SingleFlow(t *testing.T) {
	// 100 at t=1 discounted at 10% => 100/1.1
	want := 100.0 / 1.1
	if got := NPV(0.1, []float64{100}); math.Abs(got-want) > 1e-9 {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestNPV_MultiPeriod(t *testing.T) {
	// Flows: 100, 200, 300 at r=10%
	r := 0.1
	flows := []float64{100, 200, 300}
	want := 100/1.1 + 200/(1.1*1.1) + 300/(1.1*1.1*1.1)
	if got := NPV(r, flows); math.Abs(got-want) > 1e-9 {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestNPV_ZeroRate(t *testing.T) {
	flows := []float64{10, 20, 30}
	want := 60.0
	if got := NPV(0, flows); math.Abs(got-want) > 1e-9 {
		t.Fatalf("got %v want %v", got, want)
	}
}
