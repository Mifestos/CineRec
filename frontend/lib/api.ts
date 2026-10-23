import axios from 'axios';

export const api = axios.create({
  baseURL: '/api', // Используем наш rewrite прокси
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Критически важно для httpOnly cookies
});

// Интерсептор для обработки 401 (Unauthorized)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // В будущем здесь можно добавить router.push('/login')
      console.warn('Unauthorized request');
    }
    return Promise.reject(error);
  }
);