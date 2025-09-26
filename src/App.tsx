import { lazy } from 'react'

const CanvasImageLoader = lazy(() => import('./components/CanvasImageLoaderV2'))

function App() {

  return (
    <>
    <CanvasImageLoader/>
    </>
  )
}

export default App
