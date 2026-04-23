package main

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os/signal"
	"syscall"
	"time"

	finmath "analystagent/backend/internal/math"

	"go.uber.org/zap"
)

type npvRequest struct {
	CashFlows    []float64 `json:"cash_flows"`
	DiscountRate float64   `json:"discount_rate"`
}

type npvResponse struct {
	NPV float64 `json:"npv"`
}

type errResponse struct {
	Error string `json:"error"`
}

func main() {
	logger, err := zap.NewProduction()
	if err != nil {
		log.Fatalf("zap: %v", err)
	}
	defer func() { _ = logger.Sync() }()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("POST /npv", func(w http.ResponseWriter, r *http.Request) {
		if r.Body == nil {
			writeErr(w, logger, http.StatusBadRequest, "empty body")
			return
		}
		defer func() { _ = r.Body.Close() }()

		var req npvRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeErr(w, logger, http.StatusBadRequest, "invalid JSON")
			return
		}
		if req.DiscountRate <= -1 {
			writeErr(w, logger, http.StatusBadRequest, "discount_rate must be greater than -1")
			return
		}

		npv := finmath.NPV(req.DiscountRate, req.CashFlows)
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(npvResponse{NPV: npv}); err != nil {
			logger.Error("encode npv", zap.Error(err))
		}
	})

	srv := &http.Server{
		Addr:              ":8080",
		Handler:           loggingMiddleware(logger, cors(mux)),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      10 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	go func() {
		logger.Info("listening", zap.String("addr", srv.Addr))
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Fatal("listen", zap.Error(err))
		}
	}()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Error("shutdown", zap.Error(err))
	}
}

func cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func loggingMiddleware(logger *zap.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		logger.Info("request",
			zap.String("method", r.Method),
			zap.String("path", r.URL.Path),
			zap.Duration("duration", time.Since(start)),
		)
	})
}

func writeErr(w http.ResponseWriter, logger *zap.Logger, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(errResponse{Error: msg}); err != nil {
		logger.Error("encode error", zap.Error(err))
	}
}
