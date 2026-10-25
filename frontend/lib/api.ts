import axios from 'axios';

export const api = axios.create({
  baseURL: '/api', 
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, 
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