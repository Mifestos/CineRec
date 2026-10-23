"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import axios from "axios";

import { api } from "@/lib/api";
import { useAuthStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";

// Схема валидации (зеркало Pydantic-схемы)
const loginSchema = z.object({
  username: z.string().min(3, "Минимум 3 символа"),
  password: z.string().min(8, "Минимум 8 символов"),
});

type LoginFormValues = z.infer<typeof loginSchema>;

export default function LoginPage() {
  const router = useRouter();
  const setUser = useAuthStore((state) => state.setUser);
  const [error, setError] = useState<string | null>(null);

  const { register, handleSubmit, formState: { errors, isSubmitting } } = useForm<LoginFormValues>({
    resolver: zodResolver(loginSchema),
  });

  const onSubmit = async (data: LoginFormValues) => {
    setError(null);
    try {
      // FastAPI OAuth2 требует form-urlencoded
      const formData = new URLSearchParams();
      formData.append("username", data.username);
      formData.append("password", data.password);

      const response = await api.post("/auth/login", formData, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      });

      // После успешного логина бэкенд установил httpOnly cookie.
      // Запрашиваем профиль пользователя, чтобы сохранить его в Zustand
      const userResponse = await api.get("/auth/me");
      setUser(userResponse.data);
      
      router.push("/recommendations");
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && err.response?.status === 401) {
        setError("Неверный логин или пароль");
      } else {
        setError("Произошла ошибка при подключении к серверу");
      }
    }
  };

  return (
    <div className="flex h-screen w-full items-center justify-center px-4">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle className="text-2xl">Вход в CineRec</CardTitle>
          <CardDescription>
            Введите логин и пароль для доступа к рекомендациям.
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit(onSubmit)}>
          <CardContent className="space-y-4">
            {error && <div className="text-sm font-medium text-destructive">{error}</div>}
            <div className="space-y-2">
              <Label htmlFor="username">Логин</Label>
              <Input id="username" {...register("username")} />
              {errors.username && <p className="text-xs text-destructive">{errors.username.message}</p>}
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Пароль</Label>
              <Input id="password" type="password" {...register("password")} />
              {errors.password && <p className="text-xs text-destructive">{errors.password.message}</p>}
            </div>
          </CardContent>
          <CardFooter className="flex flex-col gap-2">
            <Button type="submit" className="w-full" disabled={isSubmitting}>
              {isSubmitting ? "Вход..." : "Войти"}
            </Button>
            <Button
              type="button"
              variant="ghost"
              className="w-full"
              onClick={() => router.push("/register")}
            >
              Нет аккаунта? Регистрация
            </Button>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}