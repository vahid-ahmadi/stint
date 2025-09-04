import { useEffect } from 'react';

const useKeyPress = (keys: string[], handler: (event: KeyboardEvent) => void) => {
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (keys.includes(event.key)) {
        handler(event);
      }
    };

    window.addEventListener('keydown', handleKeyPress);

    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [keys, handler]);
};

export default useKeyPress;