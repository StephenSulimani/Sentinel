import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FileText, Loader2, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useAuth } from "@/contexts/AuthContext";
import { aiUrl } from "@/lib/config";

type SavedListItem = {
  id: number;
  ticker: string;
  company_name: string;
  created_at: string;
};

export function LibraryPage() {
  const { accessToken, user, bootstrapping } = useAuth();
  const navigate = useNavigate();
  const [items, setItems] = useState<SavedListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const load = useCallback(async () => {
    if (!accessToken) {
      setItems([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError("");
    const res = await fetch(`${aiUrl}/api/saved-reports`, {
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    const text = await res.text();
    if (!res.ok) {
      let msg = text.slice(0, 400) || `Could not load library (${res.status})`;
      try {
        const j = JSON.parse(text) as { error?: string };
        if (typeof j.error === "string" && j.error.trim()) msg = j.error;
      } catch {
        /* keep */
      }
      setError(msg);
      setItems([]);
      setLoading(false);
      return;
    }
    try {
      const j = JSON.parse(text) as { items?: SavedListItem[] };
      setItems(Array.isArray(j.items) ? j.items : []);
    } catch {
      setError("Unexpected response from server.");
      setItems([]);
    }
    setLoading(false);
  }, [accessToken]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!bootstrapping && !user) {
      navigate("/login", { replace: true, state: { from: "/library" } });
    }
  }, [bootstrapping, user, navigate]);

  const remove = (id: number) => {
    if (!accessToken) return;
    setDeletingId(id);
    void (async () => {
      const res = await fetch(`${aiUrl}/api/saved-reports/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      setDeletingId(null);
      if (res.ok) setItems((prev) => prev.filter((x) => x.id !== id));
    })();
  };

  if (!user && bootstrapping) {
    return (
      <div className="flex flex-1 items-center justify-center py-24 text-muted-foreground">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (!user) return null;

  return (
    <div className="flex-1 bg-[radial-gradient(ellipse_at_top,_hsl(152_76%_42%/0.1),_transparent_50%),_hsl(224_71%_4%)] px-4 py-10">
      <div className="mx-auto max-w-3xl">
        <div className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight text-foreground">Your library</h1>
          <p className="mt-2 text-muted-foreground">
            Reports you generate while signed in are saved here automatically. Open any note to pick up
            where you left off.
          </p>
        </div>

        <Card className="border-border/80 bg-card/70 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <FileText className="h-5 w-5 text-primary" />
              Saved research
            </CardTitle>
            <CardDescription>Most recent first (up to 200 items).</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {loading ? (
              <p className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading…
              </p>
            ) : null}
            {error ? (
              <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                {error}
              </p>
            ) : null}
            {!loading && !error && items.length === 0 ? (
              <p className="rounded-md border border-dashed border-border/80 bg-muted/10 px-4 py-8 text-center text-sm text-muted-foreground">
                No saved reports yet.{" "}
                <Link to="/research" className="text-primary underline-offset-4 hover:underline">
                  Run your first ticker
                </Link>
                .
              </p>
            ) : null}
            <ul className="space-y-2">
              {items.map((it) => (
                <li
                  key={it.id}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border/70 bg-background/40 px-3 py-3"
                >
                  <div className="min-w-0">
                    <Link
                      to={`/saved/${it.id}`}
                      className="font-medium text-foreground hover:text-primary hover:underline"
                    >
                      {it.company_name}{" "}
                      <span className="font-mono text-primary">({it.ticker})</span>
                    </Link>
                    <p className="mt-0.5 font-mono text-[10px] text-muted-foreground">{it.created_at}</p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <Button size="sm" variant="secondary" asChild>
                      <Link to={`/saved/${it.id}`}>Open</Link>
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      type="button"
                      className="text-destructive hover:bg-destructive/10"
                      disabled={deletingId === it.id}
                      onClick={() => remove(it.id)}
                      title="Delete from library"
                    >
                      {deletingId === it.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
