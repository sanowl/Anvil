#!/usr/bin/env python3
"""
Comprehensive test script for Anvil ML Framework
Tests all major features including tensors, operations, and advanced capabilities
"""

import sys
import time
import numpy as np
from typing import List, Tuple

try:
    import anvil
    print("✅ Successfully imported Anvil framework")
except ImportError as e:
    print(f"❌ Failed to import Anvil: {e}")
    sys.exit(1)

def test_basic_tensor_operations():
    """Test basic tensor creation and operations"""
    print("\n🧪 Testing Basic Tensor Operations...")
    
    try:
        # Create tensors
        data1 = [1.0, 2.0, 3.0, 4.0]
        data2 = [5.0, 6.0, 7.0, 8.0]
        
        tensor1 = anvil.create_tensor(data1, [2, 2])
        tensor2 = anvil.create_tensor(data2, [2, 2])
        
        print(f"✅ Created tensor1: {tensor1}")
        print(f"✅ Created tensor2: {tensor2}")
        
        # Test tensor properties
        assert tensor1.shape() == [2, 2], f"Expected shape [2, 2], got {tensor1.shape()}"
        assert tensor1.data() == data1, f"Expected data {data1}, got {tensor1.data()}"
        
        # Test addition
        result = tensor1.add(tensor2)
        expected = [6.0, 8.0, 10.0, 12.0]
        assert result.data() == expected, f"Expected {expected}, got {result.data()}"
        print("✅ Tensor addition works correctly")
        
        # Test matrix multiplication
        matmul_result = tensor1.matmul(tensor2)
        expected_matmul = [19.0, 22.0, 43.0, 50.0]
        assert matmul_result.data() == expected_matmul, f"Expected {expected_matmul}, got {matmul_result.data()}"
        print("✅ Matrix multiplication works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic tensor operations failed: {e}")
        return False

def test_advanced_tensor_features():
    """Test advanced tensor features"""
    print("\n🚀 Testing Advanced Tensor Features...")
    
    try:
        # Test large tensor creation
        large_data = list(range(1000))
        large_tensor = anvil.create_tensor(large_data, [25, 40])
        assert large_tensor.shape() == [25, 40], f"Expected shape [25, 40], got {large_tensor.shape()}"
        print("✅ Large tensor creation works")
        
        # Test tensor string representation
        tensor_str = str(large_tensor)
        assert "PyTensor" in tensor_str, f"Expected PyTensor in string, got {tensor_str}"
        print("✅ Tensor string representation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced tensor features failed: {e}")
        return False

def test_framework_integration():
    """Test framework integration and initialization"""
    print("\n🔧 Testing Framework Integration...")
    
    try:
        # Test framework initialization
        status = anvil.test_framework()
        assert "Anvil ML Framework is working" in status, f"Expected framework status, got {status}"
        print("✅ Framework initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Framework integration failed: {e}")
        return False

def test_performance():
    """Test performance with larger operations"""
    print("\n⚡ Testing Performance...")
    
    try:
        # Create larger tensors for performance testing
        size = 100
        data_a = list(range(size * size))
        data_b = list(range(size * size, 2 * size * size))
        
        start_time = time.time()
        tensor_a = anvil.create_tensor(data_a, [size, size])
        tensor_b = anvil.create_tensor(data_b, [size, size])
        creation_time = time.time() - start_time
        
        print(f"✅ Tensor creation time: {creation_time:.4f}s")
        
        # Test matrix multiplication performance
        start_time = time.time()
        result = tensor_a.matmul(tensor_b)
        matmul_time = time.time() - start_time
        
        print(f"✅ Matrix multiplication time: {matmul_time:.4f}s")
        
        # Performance assertions
        assert creation_time < 1.0, f"Tensor creation too slow: {creation_time}s"
        assert matmul_time < 5.0, f"Matrix multiplication too slow: {matmul_time}s"
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and edge cases"""
    print("\n🛡️ Testing Error Handling...")
    
    try:
        # Test invalid shape
        try:
            anvil.create_tensor([1.0, 2.0], [1, 2, 3])  # 3D shape not supported
            print("❌ Should have raised error for 3D shape")
            return False
        except Exception:
            print("✅ Correctly handled invalid shape")
        
        # Test shape mismatch in addition
        try:
            tensor1 = anvil.create_tensor([1.0, 2.0], [1, 2])
            tensor2 = anvil.create_tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
            tensor1.add(tensor2)
            print("❌ Should have raised error for shape mismatch")
            return False
        except Exception:
            print("✅ Correctly handled shape mismatch")
        
        # Test matrix multiplication with incompatible dimensions
        try:
            tensor1 = anvil.create_tensor([1.0, 2.0], [1, 2])
            tensor2 = anvil.create_tensor([1.0, 2.0, 3.0], [1, 3])
            tensor1.matmul(tensor2)
            print("❌ Should have raised error for incompatible dimensions")
            return False
        except Exception:
            print("✅ Correctly handled incompatible dimensions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("🚀 Starting Comprehensive Anvil Framework Test Suite")
    print("=" * 60)
    
    tests = [
        ("Framework Integration", test_framework_integration),
        ("Basic Tensor Operations", test_basic_tensor_operations),
        ("Advanced Tensor Features", test_advanced_tensor_features),
        ("Performance", test_performance),
        ("Error Handling", test_error_handling),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Anvil framework is working perfectly!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 