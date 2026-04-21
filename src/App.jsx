import InventoryPredictor from './InventoryPredictor'
import './App.css'

function App() {
  return (
    <div>
      <header style={{ padding: '20px', backgroundColor: '#f5f5f5', textAlign: 'center', borderBottom: '2px solid #1976d2' }}>
        <h1>Warehouse Inventory Management System</h1>
        <p>Dynamic ML-powered inventory reorder prediction tool</p>
      </header>
      <main>
      <InventoryPredictor />
      </main>
    </div>
  )
}

export default App
