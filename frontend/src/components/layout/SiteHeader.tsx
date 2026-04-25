import { Link, NavLink } from "react-router-dom";
import { LineChart, LogIn, LogOut, UserPlus, Library, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";

const navCls =
  "text-sm font-medium text-muted-foreground hover:text-foreground transition-colors";

export function SiteHeader() {
  const { user, bootstrapping, logout } = useAuth();

  return (
    <header className="sticky top-0 z-50 border-b border-border/80 bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
        <Link to="/" className="flex items-center gap-2 shrink-0">
          <span className="flex h-9 w-9 items-center justify-center rounded-md border border-primary/40 bg-primary/10">
            <Sparkles className="h-4 w-4 text-primary" aria-hidden />
          </span>
          <div className="leading-tight">
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              Sentinel
            </p>
            <p className="text-sm font-semibold tracking-tight text-foreground">Research</p>
          </div>
        </Link>

        <nav className="hidden items-center gap-6 sm:flex">
          <NavLink to="/" end className={({ isActive }) => (isActive ? `${navCls} text-foreground` : navCls)}>
            Home
          </NavLink>
          <NavLink
            to="/research"
            className={({ isActive }) => (isActive ? `${navCls} text-foreground` : navCls)}
          >
            Research
          </NavLink>
          {user ? (
            <NavLink
              to="/library"
              className={({ isActive }) => (isActive ? `${navCls} text-foreground` : navCls)}
            >
              Library
            </NavLink>
          ) : null}
        </nav>

        <div className="flex items-center gap-2">
          {bootstrapping ? (
            <span className="text-xs text-muted-foreground">…</span>
          ) : user ? (
            <>
              <span className="hidden max-w-[10rem] truncate text-xs text-muted-foreground sm:inline">
                {user.email}
              </span>
              <Button variant="outline" size="sm" className="gap-1" onClick={() => logout()} type="button">
                <LogOut className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Log out</span>
              </Button>
            </>
          ) : (
            <>
              <Button variant="ghost" size="sm" className="gap-1" asChild>
                <Link to="/login">
                  <LogIn className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">Log in</span>
                </Link>
              </Button>
              <Button size="sm" className="gap-1" asChild>
                <Link to="/register">
                  <UserPlus className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">Sign up</span>
                </Link>
              </Button>
            </>
          )}
        </div>
      </div>
      <div className="flex border-t border-border/60 px-4 py-2 sm:hidden">
        <nav className="flex w-full justify-around gap-2 text-xs">
          <Link to="/" className="text-muted-foreground">
            Home
          </Link>
          <Link to="/research" className="flex items-center gap-1 text-muted-foreground">
            <LineChart className="h-3 w-3" />
            Research
          </Link>
          {user ? (
            <Link to="/library" className="flex items-center gap-1 text-muted-foreground">
              <Library className="h-3 w-3" />
              Library
            </Link>
          ) : null}
        </nav>
      </div>
    </header>
  );
}
