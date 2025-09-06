/**
 * Modern JavaScript ES6+ examples and web components
 */

// Arrow functions and destructuring
const users = [
    { id: 1, name: 'Alice', age: 30, city: 'New York' },
    { id: 2, name: 'Bob', age: 25, city: 'San Francisco' },
    { id: 3, name: 'Charlie', age: 35, city: 'Chicago' }
];

// Higher-order functions
const getAdultUsers = (users) => users.filter(user => user.age >= 18);
const getUserNames = (users) => users.map(({ name }) => name);
const getTotalAge = (users) => users.reduce((sum, { age }) => sum + age, 0);

// Async/await with fetch
async function fetchUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const userData = await response.json();
        return userData;
    } catch (error) {
        console.error('Failed to fetch user:', error);
        throw error;
    }
}

// Promise chain
const processUsers = () => {
    return fetchUserData(1)
        .then(user => ({ ...user, processed: true }))
        .then(processedUser => {
            console.log('Processed:', processedUser);
            return processedUser;
        })
        .catch(error => {
            console.error('Processing failed:', error);
        });
};

// Custom Web Component
class UserCard extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
    }

    static get observedAttributes() {
        return ['name', 'age', 'city'];
    }

    attributeChangedCallback(name, oldValue, newValue) {
        this.render();
    }

    connectedCallback() {
        this.render();
    }

    render() {
        const name = this.getAttribute('name') || 'Unknown';
        const age = this.getAttribute('age') || 'Unknown';
        const city = this.getAttribute('city') || 'Unknown';

        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    padding: 16px;
                    margin: 8px;
                    font-family: Arial, sans-serif;
                }
                
                .user-name {
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #333;
                }
                
                .user-details {
                    color: #666;
                    margin-top: 8px;
                }
            </style>
            
            <div class="user-name">${name}</div>
            <div class="user-details">
                Age: ${age} | City: ${city}
            </div>
        `;
    }
}

// Register the custom element
customElements.define('user-card', UserCard);

// Generator function for infinite sequences
function* fibonacciGenerator() {
    let [a, b] = [0, 1];
    while (true) {
        yield a;
        [a, b] = [b, a + b];
    }
}

// Usage with iterators
const fib = fibonacciGenerator();
const first10Fib = Array.from({ length: 10 }, () => fib.next().value);
console.log('First 10 Fibonacci:', first10Fib);

// Module pattern with closures
const Counter = (() => {
    let count = 0;

    return {
        increment: () => ++count,
        decrement: () => --count,
        value: () => count,
        reset: () => count = 0
    };
})();

// Proxy for reactive data
const createReactiveObject = (obj) => {
    return new Proxy(obj, {
        set(target, property, value) {
            console.log(`Setting ${property} to ${value}`);
            target[property] = value;
            return true;
        },
        get(target, property) {
            console.log(`Getting ${property}`);
            return target[property];
        }
    });
};

// WeakMap for private properties
const privateData = new WeakMap();

class Person {
    constructor(name, ssn) {
        this.name = name;
        privateData.set(this, { ssn });
    }

    getSSN() {
        return privateData.get(this).ssn;
    }
}

export { UserCard, Counter, createReactiveObject, Person };

