package com.deep.framework.ast;

import com.deep.framework.ast.declaration.*;
import com.deep.framework.ast.expression.ArrayExpression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.expression.ParametersExpression;
import com.deep.framework.ast.expression.TypeParametersExpression;
import com.deep.framework.ast.lexer.BlockLexer;
import com.deep.framework.ast.lexer.Lexer;
import com.deep.framework.ast.lexer.StringLexer;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;
import com.deep.framework.ast.statement.Statement;
import com.deep.framework.ast.type.Type;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static com.deep.framework.ast.lexer.TokenType.*;
import static javax.lang.model.SourceVersion.isIdentifier;

public class Parser {
    {
        new Name(null);
    }

    List<TokenType> Method_Modifiers = List.of(PUBLIC, PROTECTED, PRIVATE, STATIC, FINAL, ABSTRACT, DEFAULT, SYNCHRONIZED);
    List<TokenType> Field_Modifiers = List.of(PUBLIC, PROTECTED, PRIVATE, STATIC, FINAL, VOLATILE, TRANSIENT);
    Lexer lexer = new BlockLexer(null);
    List<Object> list = new ArrayList<>();

    public void add(String a) {
        if (a.isEmpty()) return;

        TokenType type = TokenType.getType(a);
        if (Objects.nonNull(type)) {
            list.add(type);
        } else {
            list.add(a);
        }
    }

    public void parser(String strFile) {
        Stream<String> stream = FileUtil.readFile(strFile);
        stream.reduce((a, b) -> {
            if (lexer.getType().equals("string")) {
                if (a.concat(b).endsWith("\\\"")) return a.concat(b);
                if (!b.equals(StringLexer.end)) return a.concat(b);
                lexer = lexer.getLexer();
                add(a.concat(b));
                return "";
            } else if (b.equals(StringLexer.start)) {
                lexer = new StringLexer(lexer);
                add(a);
                return a.concat(b);
            } else if (Character.isWhitespace(b.charAt(0))) {
                add(a);
                return "";
            } else if (!a.isEmpty() && isIdentifier(a.concat(b))) {
                return a.concat(b);
            } else if (TokenType.startsWith(a.concat(b))) {
                return a.concat(b);
            } else {
                add(a);
                return b;
            }
        });

        CompilationUnit compilationUnit = new CompilationUnit();
        parserBlockStatement(compilationUnit);
        parserStatement(compilationUnit);
        reduce(compilationUnit);
    }

    private void parserBlockStatement(CompilationUnit compilationUnit) {
        Node node = new Node(compilationUnit);
        compilationUnit.getChildrens().add(node);
        for (Object o : list) {
            if (o.equals(TokenType.RBRACE)) {
                node = node.getPrarent();
            } else if (o.equals(TokenType.LBRACE)) {
                Node child = new BlockStatement(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(TokenType.RPAREN)) {
                node = node.getPrarent();
            } else if (o.equals(TokenType.LPAREN)) {
                Node child = new ParametersExpression(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o.equals(TokenType.RBRACK)) {
                node = node.getPrarent();
            } else if (o.equals(TokenType.LBRACK)) {
                Node child = new ArrayExpression(node);
                node.getChildrens().add(child);
                node = child;
            } else if (o instanceof String) {
                Name child = new Name((String) o);
                node.getChildrens().add(child);
            } else {
                node.getChildrens().add(o);
            }
        }
    }

    public void parserStatement(Node node) {
        List<Object> list = new ArrayList<>();
        Statement a = new Statement(node);
        for (Object n : node.getChildrens()) {
            if (n instanceof BlockStatement) {
                list.add(a);
                list.add(n);
                a = new Statement(node);
                parserStatement((Node) n);
            } else if (n.equals(TokenType.SEMI)) {
                list.add(a);
                a = new Statement(node);
            } else if (n instanceof Node) {
                parserStatement((Node) n);
                a.getChildrens().add(n);
            } else {
                a.getChildrens().add(n);
            }
        }
        if (list.isEmpty()) return;
        node.setChildrens(list);
    }

    public void reduce(Node node) {
        parserPackage(node);
        parserImport(node);
        parserClass(node);
        for (Object n : node.getChildrens()) {
            if (n instanceof Node) {
                reduce((Node) n);
            }
        }
    }

    public void parserClass(Node node) {
        List.copyOf(node.getChildrens()).stream().reduce((a, b) -> {
            if (a instanceof Node m && b instanceof BlockStatement n) {
                if (m.getChildrens().contains(CLASS)) {
                    ClassOrInterfaceDeclaration classDeclare = new ClassOrInterfaceDeclaration(node.getPrarent());
                    List<TokenType> modifiers = m.getChildrens().stream().filter(e -> Field_Modifiers.contains(e)).map(o -> (TokenType) o).collect(Collectors.toList());
                    classDeclare.setModifiers(modifiers);
                    classDeclare.setBody(n);
                    classDeclare.setChildrens(m.getChildrens());
                    classDeclare.getChildrens().add(n);

                    m.getChildrens().removeAll(modifiers);
                    m.getChildrens().remove(CLASS);
                    node.replaceAndRemove(m, classDeclare, n);

                    parserTypeParameters(classDeclare);

                    parserMethod(n);
                    parserField(n);
                }
            }
            return b;
        });
    }

    public void parserMethod(Node node) {
        if (node instanceof MethodDeclaration) return;
        List.copyOf(node.getChildrens()).stream().map(a -> (Node) a).reduce((a, b) -> {
            if (b instanceof BlockStatement) {
                Object m = a.getChildrens().get(a.getChildrens().size() - 2);
                Object n = a.getChildrens().get(a.getChildrens().size() - 1);
                if (m instanceof Name && n instanceof ParametersExpression && b instanceof BlockStatement) {
                    MethodDeclaration methodDeclare = new MethodDeclaration(node.getPrarent());
                    List<TokenType> modifiers = a.getChildrens().stream().filter(e -> Method_Modifiers.contains(e)).map(o -> (TokenType) o).collect(Collectors.toList());
                    a.getChildrens().removeAll(modifiers);

                    methodDeclare.setModifiers(modifiers);
                    methodDeclare.setName((Name) m);
                    methodDeclare.setParameters((ParametersExpression) n);
                    methodDeclare.setBody((BlockStatement) b);
                    methodDeclare.setChildrens(a.getChildrens());
                    methodDeclare.getChildrens().add(b);

                    parserTypeParameters(methodDeclare);

                    node.replaceAndRemove(a, methodDeclare, b);
                }
            }
            return b;
        });
    }

    public void parserField(Node node) {
        if (node instanceof MethodDeclaration) return;
        List.copyOf(node.getChildrens()).stream().map(a -> (Node) a).forEach((a) -> {
            if (a instanceof Statement) {
                FieldDeclaration fieldDeclare = new FieldDeclaration(node.getPrarent());
                List<TokenType> modifiers = a.getChildrens().stream().filter(e -> Field_Modifiers.contains(e)).map(o -> (TokenType) o).collect(Collectors.toList());
                a.getChildrens().removeAll(modifiers);

                fieldDeclare.setModifiers(modifiers);
                fieldDeclare.setType(new Type((Name) a.getChildrens().stream().findFirst().get()));
                fieldDeclare.setChildrens(a.getChildrens());
                node.replace(a, fieldDeclare);
            }
        });
    }

    public void parserTypeParameters(Node node) {
        if (!(node instanceof MethodDeclaration || node instanceof ClassOrInterfaceDeclaration)) return;
        if (node.getChildrens().contains(LT)) {
            TypeParametersExpression m = null;
            for (Object a : node.getChildrens()) {
                if (a.equals(GT)) {
                    node.getChildrens().remove(a);
                } else if (a.equals(LT)) {
                    m = new TypeParametersExpression(node);
                    node.replace(a, m);
                    node = m;
                } else if (Objects.nonNull(m)) {
                    m.getChildrens().add(a);
                }
            }
            node.getChildrens().removeAll(m.getChildrens());
        }
    }

    public void parserPackage(Node node) {
        if (node instanceof Statement && node.getChildrens().contains(PACKAGE)) {
            PackageDeclaration packageDeclare = new PackageDeclaration(node);
            packageDeclare.setChildrens(node.getChildrens());
            packageDeclare.getChildrens().remove(PACKAGE);
            node.getPrarent().replace(node, packageDeclare);
        }
    }

    public void parserImport(Node node) {
        if (node instanceof ImportDeclaration) return;
        ImportDeclaration importDeclaration = new ImportDeclaration(node);
        node.getChildrens().forEach(a -> {
            if (a instanceof Statement b && b.getChildrens().contains(IMPORT)) {
                importDeclaration.getChildrens().add(a);
                importDeclaration.getChildrens().remove(IMPORT);
            }
        });

        if (importDeclaration.getChildrens().isEmpty()) return;
        node.replace(importDeclaration.getChildrens().stream().findFirst().get(), importDeclaration);
        node.getChildrens().removeAll(importDeclaration.getChildrens());
    }

}