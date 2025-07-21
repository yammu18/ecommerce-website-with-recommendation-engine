/**
 * Main JavaScript file for E-Commerce Application
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Auto-hide flash messages after 5 seconds
    setTimeout(function() {
        var alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
        alerts.forEach(function(alert) {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);
    
    // Handle quantity controls in cart
    setupQuantityControls();
    
    // Handle cart form submissions with AJAX
    setupAjaxCartForms();
    
    // Toggle password visibility
    setupPasswordToggles();
});

/**
 * Set up quantity control buttons for cart items
 */
function setupQuantityControls() {
    const quantityBtns = document.querySelectorAll('.quantity-btn');
    if (quantityBtns.length === 0) return;
    
    quantityBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const action = this.getAttribute('data-action');
            const input = this.closest('.quantity-control').querySelector('.quantity-input');
            let value = parseInt(input.value);
            
            if (action === 'increase') {
                value++;
            } else if (action === 'decrease' && value > 1) {
                value--;
            }
            
            // Ensure value is within min/max
            const min = parseInt(input.getAttribute('min') || '1');
            const max = parseInt(input.getAttribute('max') || '100');
            value = Math.max(min, Math.min(value, max));
            
            input.value = value;
            
            // Trigger change event
            input.dispatchEvent(new Event('change'));
        });
    });
    
    // Handle direct input changes
    const quantityInputs = document.querySelectorAll('.quantity-input');
    quantityInputs.forEach(input => {
        input.addEventListener('change', function() {
            let value = parseInt(this.value);
            
            // Validate input
            const min = parseInt(this.getAttribute('min') || '1');
            const max = parseInt(this.getAttribute('max') || '100');
            
            if (isNaN(value) || value < min) {
                value = min;
            } else if (value > max) {
                value = max;
            }
            
            this.value = value;
            
            // If in cart, update item
            const updateUrl = this.getAttribute('data-update-url');
            if (updateUrl) {
                updateCartItem(updateUrl, value);
            }
        });
    });
}

/**
 * Update cart item with AJAX
 */
function updateCartItem(url, quantity) {
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-Requested-With': 'XMLHttpRequest'
        },
        body: `quantity=${quantity}`
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // If success, reload the page to show updated cart
            window.location.reload();
        } else {
            // Show error message
            alert(data.message || 'An error occurred');
        }
    })
    .catch(error => {
        console.error('Error updating cart:', error);
        alert('An error occurred while updating your cart. Please try again.');
    });
}

/**
 * Set up AJAX form submissions for cart actions
 */
function setupAjaxCartForms() {
    // Add to cart forms
    const addToCartForms = document.querySelectorAll('.ajax-add-to-cart');
    addToCartForms.forEach(form => {
        form.addEventListener('submit', function(event) {
            event.preventDefault();
            
            const formData = new FormData(this);
            const url = this.getAttribute('action');
            
            fetch(url, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Show success toast
                    showToast('Success', 'Product added to your cart!', 'success');
                    
                    // Update cart count in navbar
                    updateCartCount(data.cart_count);
                    
                    // Reset quantity if needed
                    const quantityInput = this.querySelector('input[name="quantity"]');
                    if (quantityInput) {
                        quantityInput.value = 1;
                    }
                } else {
                    showToast('Error', data.message || 'An error occurred', 'danger');
                }
            })
            .catch(error => {
                console.error('Error adding to cart:', error);
                showToast('Error', 'An error occurred while adding to cart. Please try again.', 'danger');
            });
        });
    });
    
    // Remove from cart forms
    const removeFromCartForms = document.querySelectorAll('.ajax-remove-from-cart');
    removeFromCartForms.forEach(form => {
        form.addEventListener('submit', function(event) {
            event.preventDefault();
            
            const url = this.getAttribute('action');
            const productName = this.getAttribute('data-product-name') || 'Item';
            
            if (confirm(`Remove ${productName} from your cart?`)) {
                fetch(url, {
                    method: 'POST',
                    headers: {
                        'X-Requested-With': 'XMLHttpRequest'
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Reload page to show updated cart
                        window.location.reload();
                    } else {
                        showToast('Error', data.message || 'An error occurred', 'danger');
                    }
                })
                .catch(error => {
                    console.error('Error removing from cart:', error);
                    showToast('Error', 'An error occurred while removing from cart. Please try again.', 'danger');
                });
            }
        });
    });
}

/**
 * Toggle password visibility
 */
function setupPasswordToggles() {
    const toggleButtons = document.querySelectorAll('.password-toggle');
    toggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const passwordInput = document.getElementById(this.getAttribute('data-target'));
            if (passwordInput) {
                if (passwordInput.type === 'password') {
                    passwordInput.type = 'text';
                    this.innerHTML = '<i class="fas fa-eye-slash"></i>';
                } else {
                    passwordInput.type = 'password';
                    this.innerHTML = '<i class="fas fa-eye"></i>';
                }
            }
        });
    });
}

/**
 * Update cart count in navbar
 */
function updateCartCount(count) {
    const cartBadge = document.querySelector('.navbar .badge');
    if (cartBadge) {
        cartBadge.textContent = count;
    }
}

/**
 * Show toast notification
 */
function showToast(title, message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type}`;
    toast.setAttribute('id', toastId);
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    // Toast content
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <strong>${title}:</strong> ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    // Add toast to container
    toastContainer.appendChild(toast);
    
    // Initialize and show toast
    const bsToast = new bootstrap.Toast(toast, { autohide: true, delay: 5000 });
    bsToast.show();
    
    // Remove from DOM after hiding
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

/**
 * Format currency
 */
function formatCurrency(amount) {
    return '$' + parseFloat(amount).toFixed(2);
}

/**
 * Format date
 */
function formatDate(dateString, format = 'MM/DD/YYYY') {
    const date = new Date(dateString);
    if (isNaN(date)) return dateString;
    
    const month = date.getMonth() + 1;
    const day = date.getDate();
    const year = date.getFullYear();
    
    // Simple formatting - can be expanded for more complex formats
    return format
        .replace('MM', month.toString().padStart(2, '0'))
        .replace('DD', day.toString().padStart(2, '0'))
        .replace('YYYY', year);
}