// kernel_key.hpp: reduce a CUDA kernel's symbol name to the short form NCU prints ("--print-kernel-base function"):
// no return type, no namespaces, no template arguments, no parameter list.
//     _ZN2at6native29vectorized_elementwise_kernelILi4E...  ->  vectorized_elementwise_kernel
//     _apply_write_kernel (a Triton kernel, not mangled)    ->  _apply_write_kernel
#ifndef KERNEL_KEY_HPP
#define KERNEL_KEY_HPP

#include <cxxabi.h>
#include <cstdlib>
#include <string>
#include <vector>

inline std::string ncu_function_base(const char* symbol) {
    std::string s = symbol ? symbol : "";
    if (s.compare(0, 2, "_Z") == 0) {
        int status = 0;
        char* d = abi::__cxa_demangle(symbol, nullptr, nullptr, &status);
        if (status == 0 && d) s = d;
        std::free(d);
    }
    for (size_t p; (p = s.find("(anonymous namespace)")) != std::string::npos;)    // its parentheses would look like a parameter list
        s.replace(p, 21, "anonymous");

    // One left-to-right pass with a bracket stack. Inside (...), {...} and [...] a '<' or '>' is an operator, not a template
    // bracket, as in "std::enable_if<((8)>(0)), void>::type". match[i] = where the template argument list closed by s[i] opens.
    std::vector<long> match(s.size(), -1);
    std::vector<std::pair<char, size_t>> stack;
    size_t end = s.size();
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        const char top = stack.empty() ? 0 : stack.back().first;
        const bool in_expr = top == '(' || top == '{' || top == '[';
        if (c == '(' || c == '{' || c == '[') {
            if (c == '(' && stack.empty()) { end = i; break; }       // the function's own parameter list
            stack.push_back({c, i});
        } else if (c == ')' || c == '}' || c == ']') {
            if (!stack.empty()) stack.pop_back();
        } else if (c == '<' && !in_expr) {
            stack.push_back({c, i});
        } else if (c == '>' && top == '<') {
            match[i] = static_cast<long>(stack.back().second);
            stack.pop_back();
        }
    }
    s.resize(end);
    while (!s.empty() && s.back() == ' ') s.pop_back();
    if (!s.empty() && s.back() == '>' && match[s.size() - 1] >= 0)   // drop the function's own template arguments
        s.resize(static_cast<size_t>(match[s.size() - 1]));

    size_t sp = s.rfind(' ');                                 // "void ns::name" -> "ns::name"
    if (sp != std::string::npos) s.erase(0, sp + 1);
    size_t cc = s.rfind("::");                                // "ns::name" -> "name"
    if (cc != std::string::npos) s.erase(0, cc + 2);
    return s.empty() ? std::string(symbol ? symbol : "") : s;
}

#endif
