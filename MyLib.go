// mylib.go
package mylib

import (
	"bufio"
	"container/heap"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
)

/*
  //a := ([}int, n, n)
  //sort.Ints(a)

var (
        //scanner = bufio.NewScanner(os.Stdin)
        //writter = bufio.NewWriter(os.Stdout)
        inpt = bufio.NewReader(os.Stdin)
)

func main() {
        out := bufio.NewWriter(os.Stdout)
        defer out.Flush()
        // scanner.Split(bufio.ScanWords)
        fmt.Println("Hello World!")
}

func scan(args ...*int) {
        for _, arg := range args {
                scanner.Scan()
                *arg, _ = strconv.Atoi(scanner.Text())
        }
}

func print(args ...interface{}) {
        writter.WriteString(fmt.Sprintln(args...))
}
*/

// FastIO for efficient input/output
type FastIO struct {
	*bufio.Reader
	*bufio.Writer
	c [1]byte
}

// NewFastIO initializes FastIO
func NewFastIO(r io.Reader, w io.Writer) *FastIO {
	return &FastIO{Reader: bufio.NewReader(r), Writer: bufio.NewWriter(w)}
}

func (f *FastIO) readC() byte {
	f.Read(f.c[:])
	return f.c[0]
}

// NextInt reads the next integer from input
func (f *FastIO) NextInt() (int, error) {
	c := f.readC()
	for ; c <= ' '; c = f.readC() {
	}
	neg := false
	if c == '-' {
		neg = true
		c = f.readC()
	}
	n := 0
	for ; '0' <= c && c <= '9'; c = f.readC() {
		n = n*10 + int(c) - '0'
	}
	if neg {
		return -n, nil
	}
	return n, nil
}

// NextInts reads n integers from input
func (f *FastIO) NextInts(n int) ([]int, error) {
	a := make([]int, n)
	for i := 0; i < n; i++ {
		var err error
		a[i], err = f.NextInt()
		if err != nil {
			return nil, err
		}
	}
	return a, nil
}

// Next reads the next string from input
func (f *FastIO) Next() string {
	c := f.readC()
	for ; c <= ' '; c = f.readC() {
	}
	var s []byte
	for ; c > ' '; c = f.readC() {
		s = append(s, f.c[0])
	}
	return string(s)
}

// Println writes a line to output
func (f *FastIO) Println(a ...interface{}) {
	fmt.Fprintln(f.Writer, a...)
}

// Print writes to output
func (f *FastIO) Print(a ...interface{}) {
	fmt.Fprint(f.Writer, a...)
}

// PrintInts writes integers to output
func (f *FastIO) PrintInts(a ...int) {
	if len(a) > 0 {
		f.Print(a[0])
		for _, x := range a[1:] {
			f.Print(" ", x)
		}
	}
}

// PrintlnInts writes a line of integers to output
func (f *FastIO) PrintlnInts(a ...int) {
	f.PrintInts(a...)
	f.Println()
}

// Sieve of Eratosthenes
func SieveOfEratosthenes(n int) ([]int, []bool) {
	sieve := make([]bool, n+1)
	ans := make([]int, 0)
	for i := 2; i < len(sieve); i++ {
		if !sieve[i] {
			ans = append(ans, i)
			for j := i + i; j < len(sieve); j += i {
				sieve[j] = true
			}
		}
	}
	return ans, sieve
}

// MinMax returns the minimum and maximum values in ns
func MinMax(ns ...int) (int, int) {
	if len(ns) == 0 {
		return 0, 0 // Handle empty case
	}
	m, M := ns[0], ns[0]
	for _, n := range ns[1:] {
		if n < m {
			m = n
		}
		if n > M {
			M = n
		}
	}
	return m, M
}

// Min returns the smaller of two integers
func Min(i, j int) int {
	if i < j {
		return i
	}
	return j
}

// Max returns the larger of two integers
func Max(i, j int) int {
	if i > j {
		return i
	}
	return j
}
func readInt(inpt *bufio.Reader) int {
	l, _ := strconv.Atoi(readLine(inpt))
	return l
}

func readLine(inpt *bufio.Reader) string {
	l, _ := inpt.ReadString('\n')
	return strings.TrimSpace(l)
}

func readArrString(inpt *bufio.Reader) []string {
	return strings.Split(readLine(inpt), " ")
}

func readArrInt(inpt *bufio.Reader) []int {
	r := readArrString(inpt)
	arr := make([]int, len(r))
	for i := 0; i < len(arr); i++ {
		arr[i], _ = strconv.Atoi(r[i])
	}
	return arr
}

func readArrInt64(inpt *bufio.Reader) []int64 {
	r := readArrString(inpt)
	arr := make([]int64, len(r))
	for i := 0; i < len(arr); i++ {
		arr[i], _ = strconv.ParseInt(r[i], 10, 64)
	}
	return arr
}

func write(arg ...interface{})            { fmt.Print(arg...) }
func writeLine(arg ...interface{})        { fmt.Println(arg...) }
func writeF(f string, arg ...interface{}) { fmt.Printf(f, arg...) }

// KMP algorithm
type KMP struct {
	pattern []rune
	lps     []int
}

// NewKMP initializes the KMP algorithm
func NewKMP(pattern []rune) *KMP {
	kmp := &KMP{pattern: pattern}
	kmp.computeLPS()
	return kmp
}

func (k *KMP) computeLPS() {
	k.lps = make([]int, len(k.pattern))
	length := 0
	i := 1
	for i < len(k.pattern) {
		if k.pattern[i] == k.pattern[length] {
			length++
			k.lps[i] = length
			i++
		} else {
			if length != 0 {
				length = k.lps[length-1]
			} else {
				k.lps[i] = 0
				i++
			}
		}
	}
}

// Search finds occurrences of the pattern in the text
func (k *KMP) Search(text []rune) []int {
	var matches []int
	i, j := 0, 0
	for i < len(text) {
		if k.pattern[j] == text[i] {
			i++
			j++
		}
		if j == len(k.pattern) {
			matches = append(matches, i-j)
			j = k.lps[j-1]
		} else if i < len(text) && k.pattern[j] != text[i] {
			if j != 0 {
				j = k.lps[j-1]
			} else {
				i++
			}
		}
	}
	return matches
}

// Ternary Search
func TernarySearch(arr []int, left, right, key int) int {
	if right >= left {
		mid1 := left + (right-left)/3
		mid2 := right - (right-left)/3

		if arr[mid1] == key {
			return mid1
		}
		if arr[mid2] == key {
			return mid2
		}

		if key < arr[mid1] {
			return TernarySearch(arr, left, mid1-1, key)
		} else if key > arr[mid2] {
			return TernarySearch(arr, mid2+1, right, key)
		} else {
			return TernarySearch(arr, mid1+1, mid2-1, key)
		}
	}
	return -1
}

// Suffix Array
type SuffixArray struct{}

// Build constructs the suffix array from the text
func (s *SuffixArray) Build(text string) []int {
	n := len(text)
	suffixes := make([][2]int, n)
	for i := 0; i < n; i++ {
		suffixes[i] = [2]int{i, int(text[i])}
	}

	sort.Slice(suffixes, func(i, j int) bool {
		return suffixes[i][1] < suffixes[j][1]
	})

	suffixArray := make([]int, n)
	for i := 0; i < n; i++ {
		suffixArray[i] = suffixes[i][0]
	}
	return suffixArray
}

// Heap implementation
type Heap struct {
	data []int
}

// NewHeap initializes a new heap
func NewHeap() *Heap {
	return &Heap{data: []int{}}
}

// Insert adds a value to the heap
func (h *Heap) Insert(value int) {
	h.data = append(h.data, value)
	h.heapifyUp(len(h.data) - 1)
}

// Extract removes and returns the top value from the heap
func (h *Heap) Extract() (int, error) {
	if len(h.data) == 0 {
		return 0, fmt.Errorf("heap is empty")
	}
	top := h.data[0]
	h.data[0] = h.data[len(h.data)-1]
	h.data = h.data[:len(h.data)-1]
	h.heapifyDown(0)
	return top, nil
}

// heapifyUp ensures the heap property is maintained after insertion
func (h *Heap) heapifyUp(index int) {
	for index > 0 && h.data[index] < h.data[h.parent(index)] {
		h.data[index], h.data[h.parent(index)] = h.data[h.parent(index)], h.data[index]
		index = h.parent(index)
	}
}

// heapifyDown ensures the heap property is maintained after extraction
func (h *Heap) heapifyDown(index int) {
	smallest := index
	left := h.left(index)
	right := h.right(index)

	if left < len(h.data) && h.data[left] < h.data[smallest] {
		smallest = left
	}
	if right < len(h.data) && h.data[right] < h.data[smallest] {
		smallest = right
	}
	if smallest != index {
		h.data[index], h.data[smallest] = h.data[smallest], h.data[index]
		h.heapifyDown(smallest)
	}
}

func (h *Heap) parent(index int) int {
	return (index - 1) / 2
}

func (h *Heap) left(index int) int {
	return 2*index + 1
}

func (h *Heap) right(index int) int {
	return 2*index + 2
}

// Binary Tree implementation
type TreeNode struct {
	value int
	left  *TreeNode
	right *TreeNode
}

type BinaryTree struct {
	root *TreeNode
}

// Insert adds a value to the binary tree
func (t *BinaryTree) Insert(value int) {
	t.root = t.insert(t.root, value)
}

func (t *BinaryTree) insert(node *TreeNode, value int) *TreeNode {
	if node == nil {
		return &TreeNode{value: value}
	}
	if value < node.value {
		node.left = t.insert(node.left, value)
	} else {
		node.right = t.insert(node.right, value)
	}
	return node
}

// InOrderTraversal returns the in-order traversal of the tree
func (t *BinaryTree) InOrderTraversal() []int {
	var result []int
	t.inOrder(t.root, &result)
	return result
}

func (t *BinaryTree) inOrder(node *TreeNode, result *[]int) {
	if node != nil {
		t.inOrder(node.left, result)
		*result = append(*result, node.value)
		t.inOrder(node.right, result)
	}
}

// CalculateHeight returns the height of the binary tree
func (t *BinaryTree) CalculateHeight() int {
	return t.calculateHeight(t.root)
}

func (t *BinaryTree) calculateHeight(node *TreeNode) int {
	if node == nil {
		return -1
	}
	leftHeight := t.calculateHeight(node.left)
	rightHeight := t.calculateHeight(node.right)
	return int(math.Max(float64(leftHeight), float64(rightHeight))) + 1
}

// Segment Tree implementation
type SegmentTree struct {
	tree     []int
	n        int
	treeType string
}

// NewSegmentTree initializes a new segment tree
func NewSegmentTree(nums []int, treeType string) *SegmentTree {
	n := len(nums)
	tree := &SegmentTree{tree: make([]int, 4*n), n: n, treeType: treeType}
	tree.build(nums, 1, 0, n-1)
	return tree
}

func (t *SegmentTree) build(nums []int, node, left, right int) {
	if left == right {
		t.tree[node] = nums[left]
	} else {
		mid := (left + right) / 2
		t.build(nums, 2*node, left, mid)
		t.build(nums, 2*node+1, mid+1, right)
		t.tree[node] = t.combine(t.tree[2*node], t.tree[2*node+1])
	}
}

// combine merges two values based on the tree type
func (t *SegmentTree) combine(a, b int) int {
	switch t.treeType {
	case "max":
		return int(math.Max(float64(a), float64(b)))
	case "min":
		return int(math.Min(float64(a), float64(b)))
	case "sum":
		return a + b
	}
	return 0
}

// Query retrieves the value for the specified range
func (t *SegmentTree) Query(left, right int) int {
	return t.query(1, 0, t.n-1, left, right)
}

func (t *SegmentTree) query(node, segStart, segEnd, qStart, qEnd int) int {
	if qStart > segEnd || qEnd < segStart {
		return 0 // Modify this for min/max queries
	}
	if qStart <= segStart && qEnd >= segEnd {
		return t.tree[node]
	}
	mid := (segStart + segEnd) / 2
	return t.combine(t.query(2*node, segStart, mid, qStart, qEnd),
		t.query(2*node+1, mid+1, segEnd, qStart, qEnd))
}

// Graph structure
type Graph struct {
	adjList map[int]map[int]int
}

// NewGraph initializes a new graph
func NewGraph() *Graph {
	return &Graph{adjList: make(map[int]map[int]int)}
}

// AddEdge adds an edge to the graph
func (g *Graph) AddEdge(u, v, weight int) {
	if g.adjList[u] == nil {
		g.adjList[u] = make(map[int]int)
	}
	if g.adjList[v] == nil {
		g.adjList[v] = make(map[int]int)
	}
	g.adjList[u][v] = weight
	g.adjList[v][u] = weight // For undirected graph
}

// PrimMST finds and prints the edges of the Minimum Spanning Tree using Prim's algorithm
func (g *Graph) PrimMST(start int) {
	V := len(g.adjList)
	key := make([]int, V)
	inMST := make([]bool, V)
	parent := make([]int, V)

	for i := range key {
		key[i] = math.MaxInt32
	}
	key[start] = 0
	parent[start] = -1

	pq := &PriorityQueue{}
	heap.Init(pq)
	heap.Push(pq, &Item{value: start, priority: 0})

	for pq.Len() > 0 {
		u := heap.Pop(pq).(*Item).value
		inMST[u] = true

		for v, weight := range g.adjList[u] {
			if !inMST[v] && weight < key[v] {
				key[v] = weight
				parent[v] = u
				heap.Push(pq, &Item{value: v, priority: key[v]})
			}
		}
	}

	for i := 1; i < V; i++ {
		fmt.Printf("%d - %d\n", parent[i], i)
	}
}

// KruskalMST finds and prints the edges of the Minimum Spanning Tree using Kruskal's algorithm
func (g *Graph) KruskalMST() {
	edges := make([]Edge, 0)

	for u, neighbors := range g.adjList {
		for v, weight := range neighbors {
			if u < v { // To avoid duplicate edges
				edges = append(edges, Edge{weight: weight, u: u, v: v})
			}
		}
	}

	sort.Slice(edges, func(i, j int) bool {
		return edges[i].weight < edges[j].weight
	})

	parent := make([]int, len(g.adjList))
	for i := range parent {
		parent[i] = i
	}

	for _, edge := range edges {
		uRoot := find(edge.u, parent)
		vRoot := find(edge.v, parent)

		if uRoot != vRoot {
			fmt.Printf("%d - %d : %d\n", edge.u, edge.v, edge.weight)
			unionSets(uRoot, vRoot, parent)
		}
	}
}

type Edge struct {
	weight int
	u, v   int
}

// Item for priority queue
type Item struct {
	value    int
	priority int
}

// PriorityQueue for Prim's algorithm
type PriorityQueue []*Item

func (pq PriorityQueue) Len() int           { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool { return pq[i].priority < pq[j].priority }
func (pq PriorityQueue) Swap(i, j int)      { pq[i], pq[j] = pq[j], pq[i] }

func (pq *PriorityQueue) Push(x interface{}) {
	item := x.(*Item)
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// find finds the root of u
func find(u int, parent []int) int {
	if parent[u] != u {
		parent[u] = find(parent[u], parent)
	}
	return parent[u]
}

// unionSets merges two sets
func unionSets(u, v int, parent []int) {
	parent[u] = v
}

// DetectCycle checks for a cycle in the graph
func (g *Graph) DetectCycle() bool {
	visited := make(map[int]bool)
	recStack := make(map[int]bool)

	for node := range g.adjList {
		if !visited[node] {
			if g.detectCycleUtil(node, visited, recStack) {
				return true
			}
		}
	}
	return false
}

func (g *Graph) detectCycleUtil(v int, visited, recStack map[int]bool) bool {
	visited[v] = true
	recStack[v] = true

	for neighbor := range g.adjList[v] {
		if !visited[neighbor] {
			if g.detectCycleUtil(neighbor, visited, recStack) {
				return true
			}
		} else if recStack[neighbor] {
			return true
		}
	}

	recStack[v] = false
	return false
}

// TopologicalSort performs a topological sort of the graph
func (g *Graph) TopologicalSort() {
	stack := []int{}
	visited := make(map[int]bool)

	for node := range g.adjList {
		if !visited[node] {
			g.topologicalSortUtil(node, visited, &stack)
		}
	}

	for len(stack) > 0 {
		fmt.Print(stack[len(stack)-1], " ")
		stack = stack[:len(stack)-1]
	}
	fmt.Println()
}

func (g *Graph) topologicalSortUtil(v int, visited map[int]bool, stack *[]int) {
	visited[v] = true

	for neighbor := range g.adjList[v] {
		if !visited[neighbor] {
			g.topologicalSortUtil(neighbor, visited, stack)
		}
	}

	*stack = append(*stack, v)
}

// DFS performs a depth-first search
func (g *Graph) DFS(start int) {
	visited := make(map[int]bool)
	g.dfsUtil(start, visited)
	fmt.Println()
}

func (g *Graph) dfsUtil(v int, visited map[int]bool) {
	visited[v] = true
	fmt.Print(v, " ")
	for neighbor := range g.adjList[v] {
		if !visited[neighbor] {
			g.dfsUtil(neighbor, visited)
		}
	}
}

// BFS performs a breadth-first search
func (g *Graph) BFS(start int) {
	visited := make(map[int]bool)
	queue := []int{start}
	visited[start] = true

	for len(queue) > 0 {
		v := queue[0]
		queue = queue[1:]
		fmt.Print(v, " ")

		for neighbor := range g.adjList[v] {
			if !visited[neighbor] {
				visited[neighbor] = true
				queue = append(queue, neighbor)
			}
		}
	}
	fmt.Println()
}

// Dijkstra's algorithm
func (g *Graph) Dijkstra(start int) {
	dist := make(map[int]int)
	for node := range g.adjList {
		dist[node] = math.MaxInt32
	}
	dist[start] = 0

	queue := []int{start}
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]

		for v, weight := range g.adjList[u] {
			if dist[u]+weight < dist[v] {
				dist[v] = dist[u] + weight
				queue = append(queue, v)
			}
		}
	}

	for node, distance := range dist {
		if distance == math.MaxInt32 {
			fmt.Printf("Distance from %d to %d is INF\n", start, node)
		} else {
			fmt.Printf("Distance from %d to %d is %d\n", start, node, distance)
		}
	}
}

// Bellman-Ford algorithm
func (g *Graph) BellmanFord(start int) {
	dist := make(map[int]int)
	for node := range g.adjList {
		dist[node] = math.MaxInt32
	}
	dist[start] = 0

	for i := 0; i < len(g.adjList)-1; i++ {
		for u := range g.adjList {
			for v, weight := range g.adjList[u] {
				if dist[u] != math.MaxInt32 && dist[u]+weight < dist[v] {
					dist[v] = dist[u] + weight
				}
			}
		}
	}

	for node, distance := range dist {
		if distance == math.MaxInt32 {
			fmt.Printf("Distance from %d to %d is INF\n", start, node)
		} else {
			fmt.Printf("Distance from %d to %d is %d\n", start, node, distance)
		}
	}
}

// Floyd-Warshall algorithm
func (g *Graph) FloydWarshall() {
	V := len(g.adjList)
	dist := make([][]int, V)
	for i := range dist {
		dist[i] = make([]int, V)
		for j := range dist[i] {
			if i == j {
				dist[i][j] = 0
			} else {
				dist[i][j] = math.MaxInt32
			}
		}
	}

	for u, neighbors := range g.adjList {
		for v, weight := range neighbors {
			dist[u][v] = weight
		}
	}

	for k := 0; k < V; k++ {
		for i := 0; i < V; i++ {
			for j := 0; j < V; j++ {
				if dist[i][k] != math.MaxInt32 && dist[k][j] != math.MaxInt32 && dist[i][k]+dist[k][j] < dist[i][j] {
					dist[i][j] = dist[i][k] + dist[k][j]
				}
			}
		}
	}

	for i := 0; i < V; i++ {
		for j := 0; j < V; j++ {
			if dist[i][j] == math.MaxInt32 {
				fmt.Print("INF ")
			} else {
				fmt.Print(dist[i][j], " ")
			}
		}
		fmt.Println()
	}
}

// Max Subarray
func MaxSubArray(arr []int) (int, int, int) {
	if len(arr) == 0 {
		return 0, -1, -1
	}

	maxSoFar := arr[0]
	maxEndingHere := arr[0]
	start, end, startIndex := 0, 0, 0

	for i := 1; i < len(arr); i++ {
		if maxEndingHere+arr[i] > arr[i] {
			maxEndingHere += arr[i]
		} else {
			maxEndingHere = arr[i]
			startIndex = i
		}

		if maxSoFar < maxEndingHere {
			maxSoFar = maxEndingHere
			start = startIndex
			end = i
		}
	}

	return maxSoFar, start, end
}

func QuickSort(arr []int) []int {
	if len(arr) < 2 {
		return arr
	}
	pivot := arr[len(arr)/2]
	left := []int{}
	right := []int{}
	for _, v := range arr {
		if v < pivot {
			left = append(left, v)
		} else if v > pivot {
			right = append(right, v)
		}
	}
	return append(append(QuickSort(left), pivot), QuickSort(right)...)
}

// ReverseString reverses the input string
func ReverseString(s string) string {
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return string(runes)
}

// IsPalindrome checks if the input string is a palindrome
func IsPalindrome(s string) bool {
	return s == ReverseString(s)
}

// Fibonacci returns the Fibonacci sequence up to n
func Fibonacci(n int) []int {
	if n <= 0 {
		return []int{}
	}
	fib := make([]int, n)
	fib[0] = 0
	if n > 1 {
		fib[1] = 1
		for i := 2; i < n; i++ {
			fib[i] = fib[i-1] + fib[i-2]
		}
	}
	return fib
}

// AddMatrices adds two matrices
func AddMatrices(a, b [][]int) ([][]int, error) {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) || len(a[0]) != len(b[0]) {
		return nil, fmt.Errorf("matrices must be of the same size")
	}
	rows, cols := len(a), len(a[0])
	result := make([][]int, rows)
	for i := range result {
		result[i] = make([]int, cols)
		for j := range result[i] {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result, nil
}

// SubtractMatrices subtracts matrix b from matrix a
func SubtractMatrices(a, b [][]int) ([][]int, error) {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) || len(a[0]) != len(b[0]) {
		return nil, fmt.Errorf("matrices must be of the same size")
	}
	rows, cols := len(a), len(a[0])
	result := make([][]int, rows)
	for i := range result {
		result[i] = make([]int, cols)
		for j := range result[i] {
			result[i][j] = a[i][j] - b[i][j]
		}
	}
	return result, nil
}

// MultiplyMatrices multiplies two matrices
func MultiplyMatrices(a, b [][]int) ([][]int, error) {
	if len(a) == 0 || len(b) == 0 || len(a[0]) != len(b) {
		return nil, fmt.Errorf("invalid matrix dimensions for multiplication")
	}
	rowsA := len(a)
	colsB := len(b[0])
	result := make([][]int, rowsA)

	for i := range result {
		result[i] = make([]int, colsB)
		for j := 0; j < colsB; j++ {
			for k := 0; k < len(b); k++ { // Use len(b) for the number of rows in b
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result, nil
}

// Expermint not learn it yet

type Color bool

const (
	Red   Color = true
	Black Color = false
)

type RBNode struct {
	value  int
	color  Color
	left   *RBNode
	right  *RBNode
	parent *RBNode
}

type RedBlackTree struct {
	root *RBNode
}

// NewRedBlackTree initializes a new Red-Black Tree
func NewRedBlackTree() *RedBlackTree {
	return &RedBlackTree{}
}

// Insert adds a value to the Red-Black Tree
func (t *RedBlackTree) Insert(value int) {
	newNode := &RBNode{value: value, color: Red}
	t.root = t.insert(t.root, newNode)
	t.fixInsert(newNode)
}

// insert is a helper function to insert a node
func (t *RedBlackTree) insert(root, node *RBNode) *RBNode {
	if root == nil {
		return node
	}
	if node.value < root.value {
		root.left = t.insert(root.left, node)
		root.left.parent = root
	} else {
		root.right = t.insert(root.right, node)
		root.right.parent = root
	}
	return root
}

// fixInsert fixes the Red-Black Tree properties after insertion
func (t *RedBlackTree) fixInsert(node *RBNode) {
	for node != nil && node != t.root && node.parent.color == Red {
		if node.parent == node.parent.parent.left {
			uncle := node.parent.parent.right
			if uncle != nil && uncle.color == Red {
				node.parent.color = Black
				uncle.color = Black
				node.parent.parent.color = Red
				node = node.parent.parent
			} else {
				if node == node.parent.right {
					node = node.parent
					t.rotateLeft(node)
				}
				node.parent.color = Black
				node.parent.parent.color = Red
				t.rotateRight(node.parent.parent)
			}
		} else {
			uncle := node.parent.parent.left
			if uncle != nil && uncle.color == Red {
				node.parent.color = Black
				uncle.color = Black
				node.parent.parent.color = Red
				node = node.parent.parent
			} else {
				if node == node.parent.left {
					node = node.parent
					t.rotateRight(node)
				}
				node.parent.color = Black
				node.parent.parent.color = Red
				t.rotateLeft(node.parent.parent)
			}
		}
	}
	t.root.color = Black
}

// rotateLeft performs a left rotation
func (t *RedBlackTree) rotateLeft(node *RBNode) {
	y := node.right
	node.right = y.left
	if y.left != nil {
		y.left.parent = node
	}
	y.parent = node.parent
	if node.parent == nil {
		t.root = y
	} else if node == node.parent.left {
		node.parent.left = y
	} else {
		node.parent.right = y
	}
	y.left = node
	node.parent = y
}

// rotateRight performs a right rotation
func (t *RedBlackTree) rotateRight(node *RBNode) {
	y := node.left
	node.left = y.right
	if y.right != nil {
		y.right.parent = node
	}
	y.parent = node.parent
	if node.parent == nil {
		t.root = y
	} else if node == node.parent.right {
		node.parent.right = y
	} else {
		node.parent.left = y
	}
	y.right = node
	node.parent = y
}

// InOrderTraversal performs an in-order traversal of the tree
func (t *RedBlackTree) InOrderTraversal(node *RBNode, visit func(int)) {
	if node != nil {
		t.InOrderTraversal(node.left, visit)
		visit(node.value)
		t.InOrderTraversal(node.right, visit)
	}
}

type TrieNode struct {
	children map[rune]*TrieNode
	isEnd    bool
}

type Trie struct {
	root *TrieNode
}

// NewTrie initializes a new Trie
func NewTrie() *Trie {
	return &Trie{root: &TrieNode{children: make(map[rune]*TrieNode)}}
}

// Insert adds a word to the Trie
func (t *Trie) Insert(word string) {
	node := t.root
	for _, ch := range word {
		if _, exists := node.children[ch]; !exists {
			node.children[ch] = &TrieNode{children: make(map[rune]*TrieNode)}
		}
		node = node.children[ch]
	}
	node.isEnd = true
}

// Search checks if a word exists in the Trie
func (t *Trie) Search(word string) bool {
	node := t.root
	for _, ch := range word {
		if _, exists := node.children[ch]; !exists {
			return false
		}
		node = node.children[ch]
	}
	return node.isEnd
}

// StartsWith checks if there is a prefix in the Trie
func (t *Trie) StartsWith(prefix string) bool {
	node := t.root
	for _, ch := range prefix {
		if _, exists := node.children[ch]; !exists {
			return false
		}
		node = node.children[ch]
	}
	return true
}

// Example
/*
func main() {
	// Trie example
	trie := NewTrie()
	trie.Insert("hello")
	fmt.Println(trie.Search("hello")) // true
	fmt.Println(trie.Search("world"))  // false
	fmt.Println(trie.StartsWith("he")) // true

	// Red-Black Tree example
	rbTree := NewRedBlackTree()
	rbTree.Insert(10)
	rbTree.Insert(20)
	rbTree.Insert(15)

	fmt.Println("In-order traversal:")
	rbTree.InOrderTraversal(rbTree.root, func(value int) {
		fmt.Print(value, " ")
	})
	fmt.Println()
}
*/
