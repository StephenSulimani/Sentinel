import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { aiUrl } from "@/lib/config";

const STORAGE_KEY = "sentinel_access_token";

export type AuthUser = { id: number; email: string };

type AuthContextValue = {
  accessToken: string | null;
  user: AuthUser | null;
  bootstrapping: boolean;
  login: (email: string, password: string) => Promise<{ ok: true } | { ok: false; message: string }>;
  register: (
    email: string,
    password: string,
  ) => Promise<{ ok: true } | { ok: false; message: string }>;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [accessToken, setAccessToken] = useState<string | null>(() =>
    typeof window !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null,
  );
  const [user, setUser] = useState<AuthUser | null>(null);
  const [bootstrapping, setBootstrapping] = useState(true);

  useEffect(() => {
    if (!accessToken) {
      setUser(null);
      setBootstrapping(false);
      return;
    }
    let cancelled = false;
    setBootstrapping(true);
    void (async () => {
      const res = await fetch(`${aiUrl}/api/auth/me`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (cancelled) return;
      if (res.ok) {
        try {
          const j = (await res.json()) as { user?: AuthUser };
          if (j.user && typeof j.user.email === "string") {
            setUser(j.user);
          } else {
            setUser(null);
            localStorage.removeItem(STORAGE_KEY);
            setAccessToken(null);
          }
        } catch {
          setUser(null);
          localStorage.removeItem(STORAGE_KEY);
          setAccessToken(null);
        }
      } else {
        localStorage.removeItem(STORAGE_KEY);
        setAccessToken(null);
        setUser(null);
      }
      setBootstrapping(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [accessToken]);

  const persistToken = useCallback((token: string | null) => {
    if (token) localStorage.setItem(STORAGE_KEY, token);
    else localStorage.removeItem(STORAGE_KEY);
    setAccessToken(token);
  }, []);

  const login = useCallback(
    async (email: string, password: string) => {
      try {
        const res = await fetch(`${aiUrl}/api/auth/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });
        const text = await res.text();
        let msg = text.slice(0, 400) || `Login failed (${res.status})`;
        try {
          const j = JSON.parse(text) as {
            access_token?: string;
            user?: AuthUser;
            error?: string;
          };
          if (typeof j.error === "string" && j.error.trim()) msg = j.error;
          if (res.ok && j.access_token && j.user) {
            persistToken(j.access_token);
            setUser(j.user);
            return { ok: true as const };
          }
        } catch {
          /* use msg */
        }
        if (res.ok) return { ok: false as const, message: "Unexpected login response" };
        return { ok: false as const, message: msg };
      } catch (e) {
        return {
          ok: false as const,
          message: e instanceof Error ? e.message : "Login request failed",
        };
      }
    },
    [persistToken],
  );

  const register = useCallback(
    async (email: string, password: string) => {
      try {
        const res = await fetch(`${aiUrl}/api/auth/register`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });
        const text = await res.text();
        let msg = text.slice(0, 400) || `Registration failed (${res.status})`;
        try {
          const j = JSON.parse(text) as {
            access_token?: string;
            user?: AuthUser;
            error?: string;
          };
          if (typeof j.error === "string" && j.error.trim()) msg = j.error;
          if (res.ok && j.access_token && j.user) {
            persistToken(j.access_token);
            setUser(j.user);
            return { ok: true as const };
          }
        } catch {
          /* use msg */
        }
        if (res.ok) return { ok: false as const, message: "Unexpected registration response" };
        return { ok: false as const, message: msg };
      } catch (e) {
        return {
          ok: false as const,
          message: e instanceof Error ? e.message : "Registration request failed",
        };
      }
    },
    [persistToken],
  );

  const logout = useCallback(() => {
    persistToken(null);
    setUser(null);
  }, [persistToken]);

  const value = useMemo(
    () => ({
      accessToken,
      user,
      bootstrapping,
      login,
      register,
      logout,
    }),
    [accessToken, user, bootstrapping, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
