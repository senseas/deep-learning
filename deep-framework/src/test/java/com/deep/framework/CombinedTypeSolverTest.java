package com.deep.framework;

import com.deep.framework.framework.TensorCompiler;
import com.deep.framework.framework.TensorFlow;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorFunctor;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.resolution.declarations.ResolvedMethodDeclaration;
import com.github.javaparser.symbolsolver.javaparsermodel.JavaParserFacade;
import com.github.javaparser.symbolsolver.model.resolution.SymbolReference;
import com.github.javaparser.symbolsolver.model.resolution.TypeSolver;
import com.github.javaparser.symbolsolver.reflectionmodel.ReflectionMethodDeclaration;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ClassLoaderTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import lombok.SneakyThrows;

import java.io.File;
import java.io.FileNotFoundException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

public class CombinedTypeSolverTest {
    private static Map<Node, Integer> mapx = new HashMap<>();
    private static Map<String, TensorFunctor> mapf = new HashMap<>();
    private static List<String> methods = Arrays.stream(TensorFlow.class.getDeclaredMethods()).map(a -> a.toGenericString()).toList();

    static {
        TensorCompiler tc = new TensorCompiler();
        Method[] methods = tc.getClass().getDeclaredMethods();
        Arrays.stream(methods).forEach(method -> {
            try {
                Class type = (Class) method.getGenericParameterTypes()[0];
                Tensor[] args = IntStream.range(0, method.getParameterCount()).mapToObj(a -> new Tensor(new int[]{1})).toArray(Tensor[]::new);
                TensorFunctor tensor = (TensorFunctor) method.invoke(tc, type.isArray() ? new Object[]{args} : args);
                mapf.put(method.getName(), tensor);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    public static class TypeCalculatorVisitor extends VoidVisitorAdapter<JavaParserFacade> {
        @SneakyThrows
        public void visit(MethodCallExpr expr, JavaParserFacade javaParserFacade) {
            super.visit(expr, javaParserFacade);
            try {

                //ResolvedType type = javaParserFacade.getType(expr);
                //System.out.println(type.describe());

                if (expr.toString().equals("C.set(addx(C.get(l), conv(stride, padding, new Tensor(A.get(l)), new Tensor(B.get(i)))), l)")) {
                    SymbolReference<ResolvedMethodDeclaration> solve = javaParserFacade.solve(expr);
                }

                SymbolReference<ResolvedMethodDeclaration> solve = javaParserFacade.solve(expr);
                boolean solved = solve.isSolved();
                if (solved) {
                    ResolvedMethodDeclaration correspondingDeclaration = solve.getCorrespondingDeclaration();
                    if (correspondingDeclaration instanceof ReflectionMethodDeclaration) {
                        Field field = ((ReflectionMethodDeclaration) correspondingDeclaration).getClass().getDeclaredField("method");
                        field.setAccessible(true);
                        Method method = (Method) field.get(correspondingDeclaration);
                        System.out.println(method.toGenericString());
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        @Override
        public void visit(NameExpr expr, JavaParserFacade javaParserFacade) {
            super.visit(expr, javaParserFacade);
            //ResolvedType type = javaParserFacade.getType(expr);
            //System.out.println(type.describe());
            //ResolvedValueDeclaration correspondingDeclaration = javaParserFacade.solve(expr).getCorrespondingDeclaration();
            //System.out.println(correspondingDeclaration);
        }
    }

    public static void main(String[] args) throws FileNotFoundException {
        TypeSolver typeSolver = new CombinedTypeSolver(new ClassLoaderTypeSolver(CombinedTypeSolverTest.class.getClassLoader()), new ReflectionTypeSolver(), new JavaParserTypeSolver("/Users/chengdong/GitHub/deep-learning/deep-framework/src/main/java/"));
        ParseResult<CompilationUnit> parse = new JavaParser().parse(new File("/Users/chengdong/GitHub/deep-learning/deep-framework/src/main/java/com/deep/framework/framework/TensorFlow.java"));
        parse.getResult().ifPresent(a -> a.accept(new TypeCalculatorVisitor(), JavaParserFacade.get(typeSolver)));
    }

}