"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { LogOut, Star } from "lucide-react";

import { api } from "@/lib/api";
import { useAuthStore } from "@/lib/store";
import { Movie } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function RecommendationsPage() {
  const router = useRouter();
  const user = useAuthStore((state) => state.user);
  const clearUser = useAuthStore((state) => state.clearUser);

  // Защита роута: если пользователя нет в стейте, кидаем на логин
  useEffect(() => {
    if (!user) {
      router.push("/login");
    }
  }, [user, router]);

  // Запрос к бэкенду через React Query
  const { data: movies, isLoading, isError } = useQuery({
    queryKey: ["recommendations"],
    queryFn: async () => {
      const response = await api.get<Movie[]>("/v1/recommend?top_n=10");
      return response.data;
    },
    enabled: !!user, // Выполняем запрос только если пользователь авторизован
  });

  const handleLogout = () => {
    clearUser(); // Очищаем Zustand
    // В идеале здесь нужно также удалить куку на бэкенде (мы сделаем это позже)
    router.push("/login");
  };

  // Пока идет редирект, ничего не рендерим
  if (!user) return null;

  return (
    <div className="container mx-auto py-10 px-4">
      <header className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Рекомендации для вас</h1>
          <p className="text-muted-foreground mt-1">
            Привет, {user.username}. Подобрано на основе ваших оценок.
          </p>
        </div>
        <Button variant="outline" onClick={handleLogout}>
          <LogOut className="mr-2 h-4 w-4" />
          Выйти
        </Button>
      </header>

      {isLoading && (
        <div className="flex justify-center py-20">
          <p className="text-lg text-muted-foreground animate-pulse">Загрузка рекомендаций...</p>
        </div>
      )}

      {isError && (
        <div className="text-center py-20 text-destructive">
          <p>Не удалось загрузить рекомендации. Попробуйте позже.</p>
        </div>
      )}

      {movies && movies.length === 0 && (
        <div className="text-center py-20 text-muted-foreground">
          <p>Пока нет рекомендаций. Поставьте несколько оценок фильмам, чтобы мы узнали ваши вкусы.</p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {movies?.map((movie) => (
          <Card key={movie.movieId} className="flex flex-col hover:border-primary transition-colors">
            <CardHeader className="pb-3">
              <div className="flex justify-between items-start gap-2">
                <CardTitle className="text-lg leading-tight">{movie.title}</CardTitle>
                <Badge variant="secondary" className="shrink-0 flex items-center gap-1">
                  <Star className="h-3 w-3" />
                  {movie.score?.toFixed(2)}
                </Badge>
              </div>
              <CardDescription className="text-xs">
                {movie.genres.split("|").join(", ")}
              </CardDescription>
            </CardHeader>
            <CardContent className="mt-auto pt-0 flex gap-2">
              <Button 
                variant="secondary" 
                className="w-full text-xs"
                onClick={() => router.push(`/movies/${movie.movieId}`)}
              >
                Подробнее
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}